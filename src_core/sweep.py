"""Experiment driver: fixed strata -> detection sweep (model as a flag, enhancement ON/OFF) -> per-frame records
-> per-stratum metrics -> detector comparison.

    python src_core/sweep.py strata                          # build + save the seeded frame lists
    python src_core/sweep.py detect --model yolo26m          # records + metrics, enhancement off
    python src_core/sweep.py detect --model yolo26m --enhancement on
    python src_core/sweep.py detect --model yolo26m --synthetic   # + imagecorruptions severity sweep
    python src_core/sweep.py compare                         # three-detector table, picks the detector
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (DEFAULT_CONF, RESULTS, FrameGT, frame_metrics, imread, read_jsonl, record_template, save_json,  # noqa: E402
                    start_log, stratum_ap50, write_jsonl)
import danger as DG  # noqa: E402
import features as F  # noqa: E402
from enhance import enhance  # noqa: E402

ADVERSE = ["bdd_val_night", "bdd_val_rain", "bdd_val_snow", "acdc_fog", "acdc_night", "acdc_rain", "acdc_snow",
           "cs_fog_light", "cs_fog_medium", "cs_fog_dense"]
SYNTHETIC = [("fog", s) for s in (1, 3, 5)] + [("frost", s) for s in (1, 3, 5)] + [("motion_blur", s) for s in (1, 3, 5)]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("strata")
    s.add_argument("--cap-bdd-train", type=int, default=3000)
    s.add_argument("--cap-bdd-val", type=int, default=1500, help="cap for clear and night (rain and snow are all used)")
    s.add_argument("--cap-acdc", type=int, default=None)
    s.add_argument("--cap-cs-scenes", type=int, default=800, help="Cityscapes scenes; each gives clear + 3 fog frames")
    s.add_argument("--seed", type=int, default=0)
    d = sub.add_parser("detect")
    d.add_argument("--model", default="yolo26m")
    d.add_argument("--enhancement", choices=["off", "on"], default="off")
    d.add_argument("--imgsz", type=int, default=640)
    d.add_argument("--synthetic", action="store_true", help="also run the imagecorruptions severity sweep on bdd_val_clear")
    d.add_argument("--cap-synthetic", type=int, default=200)
    d.add_argument("--strata", nargs="*", default=None, help="subset of strata (default: all)")
    d.add_argument("--limit", type=int, default=None, help="frames per stratum (smoke test)")
    d.add_argument("--no-dino", action="store_true")
    d.add_argument("--batch", type=int, default=16)
    d.add_argument("--device", default="cuda")
    c = sub.add_parser("compare")
    c.add_argument("--models", nargs="+", default=["yolov8m", "yolo11m", "yolo26m"])
    c.add_argument("--enhancement", default="off")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- strata
def cmd_strata(a):
    from gt_loaders import build_strata, save_strata
    caps = {"bdd_train_clear": a.cap_bdd_train, "bdd_val_clear": a.cap_bdd_val, "bdd_val_night": a.cap_bdd_val,
            "acdc_fog": a.cap_acdc, "acdc_night": a.cap_acdc, "acdc_rain": a.cap_acdc, "acdc_snow": a.cap_acdc,
            "cs_scenes": a.cap_cs_scenes}
    strata = build_strata(caps, a.seed)
    save_strata(strata)
    rows = [{"stratum": k, "frames": len(v), "boxes": sum(len(f.boxes) for f in v),
             "persons": sum(1 for f in v for b in f.boxes if b["cls"] == "person")} for k, v in strata.items()]
    pd.DataFrame(rows).to_csv(RESULTS / "strata_counts.csv", index=False)
    print(f"[strata] saved {RESULTS / 'strata.json'}")


# --------------------------------------------------------------------------- detect
def records_path(model: str, enhancement: str) -> Path:
    return RESULTS / "records" / f"{model}_{enhancement}.jsonl"


def _load_images(frames: list[FrameGT], workers: int = 8) -> list[np.ndarray]:
    with ThreadPoolExecutor(workers) as ex:
        imgs = list(ex.map(lambda f: imread(f.path), frames))
    bad = [f.path for f, im in zip(frames, imgs) if im is None]
    if bad:
        raise RuntimeError(f"unreadable images, e.g. {bad[:3]}")
    return imgs


def _corrupt(img_bgr: np.ndarray, name: str, severity: int) -> np.ndarray:
    if not hasattr(np, "float_"):
        np.float_ = np.float64  # imagecorruptions 1.1.2 still uses the NumPy-1 alias
    from imagecorruptions import corrupt
    rgb = np.ascontiguousarray(img_bgr[:, :, ::-1])
    out = corrupt(rgb, corruption_name=name, severity=severity)
    return np.ascontiguousarray(np.asarray(out, np.uint8)[:, :, ::-1])


class SweepContext:
    def __init__(self, a):
        from detect import Detector
        from weather import WeatherModel
        self.a = a
        self.det = Detector(a.model, a.device, a.imgsz)
        self.weather = WeatherModel(device=a.device)
        self.dino = None if a.no_dino else F.DinoEmbedder(a.device)
        self.knn = None
        print(f"[detect] detector {self.det.name} on {self.det.device} imgsz {a.imgsz} | weather {self.weather.names} | dino {self.dino is not None}")

    def process(self, frames: list[FrameGT], stratum: str, corruption: dict | None = None, knn_loo: bool = False):
        """Yield one record per frame."""
        a = self.a
        for i in range(0, len(frames), a.batch):
            chunk = frames[i:i + a.batch]
            raw = _load_images(chunk)
            if corruption:
                raw = [_corrupt(im, corruption["type"], corruption["severity"]) for im in raw]
            wx = self.weather.predict(raw)
            enh = [enhance(im, w["weather"], w["fog_level"], a.enhancement == "on") for im, w in zip(raw, wx)]
            dets, dets_flip = self.det.detect_with_flip(enh)
            emb = self.dino.embed(raw) if self.dino is not None else None
            dfe = None
            if emb is not None and self.knn is not None:
                if knn_loo:  # training frames: exclude the frame itself from its own neighbours
                    d, _ = self.knn.nn.kneighbors(emb, n_neighbors=self.knn.k + 1)
                    d = d[:, 1:]
                    to_mean = 1.0 - emb @ self.knn.mean
                    dfe = [{"d_knn_dist": float(d[j].mean()), "d_knn_dist_min": float(d[j].min()), "d_norm_to_mean": float(to_mean[j])} for j in range(len(emb))]
                else:
                    dfe = self.knn.features(emb)
            for j, fg in enumerate(chunk):
                hw = raw[j].shape[:2]
                rec = record_template(fg, self.det.name, a.enhancement, corruption)
                rec["stratum"] = stratum
                rec["image_hw"] = list(hw)
                rec["detections"] = dets[j]
                rec["weather_pred"], rec["fog_level_pred"] = wx[j]["weather"], wx[j]["fog_level"]
                rec["consistency"] = F.flip_consistency(dets[j], dets_flip[j])
                meta = {}
                meta.update(F.quality_features(raw[j]))
                meta.update(F.conf_features(dets[j], DEFAULT_CONF, hw))
                meta.update(rec["consistency"])
                meta.update(F.weather_features(wx[j]["p_weather"], wx[j]["fog_level"]))
                if dfe is not None:
                    meta.update(dfe[j])
                rec["meta"] = meta
                rec["embedding"] = emb[j].astype(np.float16).tolist() if emb is not None else None
                rec["eval"] = frame_metrics(dets[j], fg.boxes)
                fog_level = fg.extra.get("fog_level", "none") if fg.weather_gt == "fog" else "none"
                g = DG.gap(dets[j], fg.boxes, hw, fg.weather_gt, fog_level)
                rec["scores"].update({"danger_est": g["danger_est"], "danger_true": g["danger_true"], "danger_gap": g["gap"]})
                yield rec


def cmd_detect(a):
    from gt_loaders import load_strata
    strata = load_strata()
    names = a.strata or list(strata)
    if a.limit:
        strata = {k: v[:a.limit] for k, v in strata.items()}
    ctx = SweepContext(a)
    out_path = records_path(ctx.det.name, a.enhancement)
    t0 = time.perf_counter()
    all_records = []
    # 1) the clear-day TRAINING stratum first: it defines the kNN reference for the feature-space distance
    if ctx.dino is not None and "bdd_train_clear" in strata:
        train = strata["bdd_train_clear"]
        print(f"[detect] embedding {len(train)} clear-day training frames for the kNN reference")
        ref = []
        for i in range(0, len(train), 64):
            ref.append(ctx.dino.embed(_load_images(train[i:i + 64])))
        ref = np.concatenate(ref)
        ctx.knn = F.KnnReference(ref, k=5)
        np.save(RESULTS / "knn_reference_bdd_train_clear.npy", ref)
    for name in names:
        frames = strata[name]
        t1 = time.perf_counter()
        n = 0
        for rec in ctx.process(frames, name, knn_loo=(name == "bdd_train_clear")):
            all_records.append(rec)
            n += 1
        print(f"[detect] {name:18s} {n:5d} frames in {time.perf_counter() - t1:6.1f}s | fail(person) "
              f"{np.mean([r['eval']['fail_person'] for r in all_records[-n:]]):.3f}")
    if a.synthetic and "bdd_val_clear" in strata:
        base = strata["bdd_val_clear"][: a.cap_synthetic]
        for ctype, sev in SYNTHETIC:
            name = f"syn_{ctype}_{sev}"
            t1 = time.perf_counter()
            n = 0
            for rec in ctx.process(base, name, corruption={"type": ctype, "severity": sev}):
                all_records.append(rec)
                n += 1
            print(f"[detect] {name:18s} {n:5d} frames in {time.perf_counter() - t1:6.1f}s | fail(person) "
                  f"{np.mean([r['eval']['fail_person'] for r in all_records[-n:]]):.3f}")
    write_jsonl(out_path, all_records)
    print(f"[detect] wrote {len(all_records)} records -> {out_path} in {(time.perf_counter() - t0) / 60:.1f} min")
    metrics = stratum_metrics(all_records)
    mp = RESULTS / f"metrics_{ctx.det.name}_{a.enhancement}.csv"
    metrics.to_csv(mp, index=False)
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"[detect] wrote {mp}")


def stratum_metrics(records: list[dict]) -> pd.DataFrame:
    rows = []
    by = {}
    for r in records:
        by.setdefault(r["stratum"], []).append(r)
    for name, recs in by.items():
        ev = [r["eval"] for r in recs]
        ap = stratum_ap50([(r["detections"], r["gt_boxes"]) for r in recs])
        tp_conf, fp_conf = [], []
        for r in recs:
            from common import filter_boxes, greedy_match, MIN_BOX_PX
            d = filter_boxes(r["detections"], DEFAULT_CONF, MIN_BOX_PX)
            g = filter_boxes(r["gt_boxes"], None, MIN_BOX_PX)
            matched = {i for i, _, _ in greedy_match(d, g)}
            for i, x in enumerate(d):
                (tp_conf if i in matched else fp_conf).append(x["conf"])
        n_gt = sum(e["n_gt"] for e in ev)
        rows.append({"stratum": name, "frames": len(recs), "map50": ap["map50"], "ap50_person": ap["per_class"].get("person", float("nan")),
                     "precision": sum(e["tp"] for e in ev) / max(sum(e["n_det"] for e in ev), 1),
                     "recall": sum(e["tp"] for e in ev) / max(n_gt, 1),
                     "person_recall": 1 - sum(e["missed_person"] for e in ev) / max(sum(e["n_gt_person"] for e in ev), 1),
                     "fail_rate_person": float(np.mean([e["fail_person"] for e in ev])),
                     "fail_rate_f1": float(np.mean([e["fail_f1"] for e in ev])),
                     "fail_rate": float(np.mean([e["fail"] for e in ev])),
                     "mean_conf_tp": float(np.mean(tp_conf)) if tp_conf else float("nan"),
                     "mean_conf_fp": float(np.mean(fp_conf)) if fp_conf else float("nan"),
                     "danger_gap_mean": float(np.mean([r["scores"]["danger_gap"] for r in recs])),
                     "weather_pred_agree": float(np.mean([r["weather_pred"] == r["weather_gt"] for r in recs]))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- compare
def cmd_compare(a):
    tables = {}
    for m in a.models:
        p = RESULTS / f"metrics_{m}_{a.enhancement}.csv"
        if p.exists():
            tables[m] = pd.read_csv(p).set_index("stratum")
        else:
            print(f"[compare] missing {p}; run detect --model {m} first")
    if not tables:
        sys.exit("[compare] nothing to compare")
    cols = ["map50", "person_recall", "precision", "fail_rate_person"]
    wide = pd.concat({m: t[cols] for m, t in tables.items()}, axis=1)
    print("[compare] per stratum:\n" + wide.to_string(float_format=lambda x: f"{x:.3f}"))
    wide.to_csv(RESULTS / f"detector_comparison_{a.enhancement}.csv")
    summary = []
    for m, t in tables.items():
        adv = t.loc[[s for s in ADVERSE if s in t.index]]
        clear = t.loc[[s for s in ("bdd_val_clear", "cs_clear") if s in t.index]]
        summary.append({"model": m, "adverse_person_recall": adv["person_recall"].mean(), "adverse_precision": adv["precision"].mean(),
                        "adverse_map50": adv["map50"].mean(), "clear_person_recall": clear["person_recall"].mean(),
                        "drop_clear_to_adverse": clear["person_recall"].mean() - adv["person_recall"].mean()})
    s = pd.DataFrame(summary).set_index("model")
    best_prec = s["adverse_precision"].max()
    eligible = s[s["adverse_precision"] >= best_prec - 0.05]  # "at equal precision": within 5 points of the best
    choice = eligible.sort_values(["adverse_person_recall", "drop_clear_to_adverse"], ascending=[False, True]).index[0]
    print("[compare] summary (adverse strata = night, rain, snow, fog of every camera):\n" + s.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"[compare] chosen detector: {choice} (best adverse pedestrian recall among models within 5 precision points of the best)")
    save_json({"detector": choice, "rule": "adverse person recall, precision within 0.05 of best, tie -> smallest clear-to-adverse drop",
               "summary": s.reset_index().to_dict("records")}, RESULTS / "detector_choice.json")


def main(argv=None):
    a = parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)
    start_log(RESULTS / f"log_sweep_{a.cmd}.txt")
    {"strata": cmd_strata, "detect": cmd_detect, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
