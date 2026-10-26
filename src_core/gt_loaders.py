"""Ground-truth box loaders for BDD100K, ACDC and Cityscapes, and the fixed, seeded stratum lists."""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

from common import (ACDC_NAME_TO_PROJECT, BDD_TO_PROJECT, CITYSCAPES_ID_TO_PROJECT, PROJECT_ROOT, RESULTS, FrameGT,
                    save_json)

sys.path.insert(0, str(PROJECT_ROOT / "src"))
import weather_data as wd  # noqa: E402  (streaming JSON reader, BDD weather mapping)

ACDC_RGB = PROJECT_ROOT / "rgb_anon_trainvaltest_ACDC" / "rgb_anon"
ACDC_DET = PROJECT_ROOT / "gt_detection_trainval" / "gt_detection"
CS_CLEAN = PROJECT_ROOT / "leftImg8bit_trainvaltest" / "leftImg8bit"
CS_FOGGY = PROJECT_ROOT / "leftImg8bit_trainval_foggyDBF" / "leftImg8bit_foggyDBF"
CS_GT = PROJECT_ROOT / "gtFine_trainvaltest" / "gtFine"
BDD_LABELS = PROJECT_ROOT / "bdd100k-10k" / "labels"
FOG_LEVELS = {"0.005": "light", "0.01": "medium", "0.02": "dense"}


def bdd_condition(attrs: dict) -> str | None:
    """clear | night | rain | snow | fog, or None for dawn/dusk and undefined (same mapping as the weather model)."""
    return wd.bdd_weather_label(attrs)


def _bdd_image_index(root: Path) -> dict[str, Path]:
    """filename -> path over the full 100k set (nested folders) and the 10k subset."""
    index = {}
    for split in ("train", "val", "test"):
        d = root / "bdd100k-10k" / split
        if d.is_dir():
            for fn in os.listdir(d):
                index[fn] = d / fn
    sys.path.insert(0, str(root / "src_v4"))
    import weather_v4_lib as L4  # noqa: E402
    big = L4.find_bdd100k_images(root)
    if big:
        for split in ("train", "val"):
            for dirpath, _, names in os.walk(big / split):
                for fn in names:
                    if fn.lower().endswith(".jpg"):
                        index[fn] = Path(dirpath) / fn
    return index


def load_bdd(label_file: Path, index: dict[str, Path], split: str) -> list[FrameGT]:
    out = []
    for rec in wd._iter_json_records(label_file):
        name = Path(str(rec.get("name", ""))).name
        cond = bdd_condition(rec.get("attributes") or {})
        if not name or cond is None or name not in index:
            continue
        boxes = []
        for l in rec.get("labels") or []:
            b = l.get("box2d")
            cls = BDD_TO_PROJECT.get(l.get("category"))
            if b and cls:
                boxes.append({"cls": cls, "xyxy": [float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])]})
        a = rec["attributes"]
        out.append(FrameGT(Path(name).stem, str(index[name]), "bdd", split, cond, str(a.get("timeofday")),
                           f"bdd_{split}_{cond}", boxes, {"weather_tag": a.get("weather"), "scene": a.get("scene"),
                                                          "video": Path(name).stem.split("-")[0]}))
    return out


def load_acdc(condition: str, split: str) -> list[FrameGT]:
    f = ACDC_DET / condition / f"instancesonly_{condition}_{split}_gt_detection.json"
    d = json.load(open(f, encoding="utf-8"))
    id2name = {c["id"]: c["name"] for c in d["categories"]}
    images = {im["id"]: im for im in d["images"] if im["file_name"].startswith(condition + "/")}  # rain-train quirk
    boxes = {i: [] for i in images}
    for a in d["annotations"]:
        if a["image_id"] in boxes and not a.get("iscrowd", 0):
            cls = ACDC_NAME_TO_PROJECT.get(id2name[a["category_id"]])
            if cls:
                x, y, w, h = a["bbox"]
                boxes[a["image_id"]].append({"cls": cls, "xyxy": [float(x), float(y), float(x + w), float(y + h)]})
    out = []
    for i, im in images.items():
        p = ACDC_RGB / im["file_name"]
        seq = Path(im["file_name"]).stem.split("_frame_")[0]
        out.append(FrameGT(Path(im["file_name"]).stem, str(p), "acdc", split, condition, "night" if condition == "night" else "daytime",
                           f"acdc_{condition}", boxes[i], {"sequence": seq}))
    return out


_CS_RE = re.compile(r"^(?P<city>[a-z]+)_(?P<seq>\d+)_(?P<frame>\d+)_gtFine_instanceIds\.png$")


def cityscapes_boxes(instance_png: Path) -> list:
    import cv2
    m = cv2.imread(str(instance_png), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise RuntimeError(f"cannot read {instance_png}")
    out = []
    for inst in np.unique(m):
        if inst < 1000:
            continue
        cls = CITYSCAPES_ID_TO_PROJECT.get(int(inst) // 1000)
        if cls is None:
            continue
        ys, xs = np.where(m == inst)
        out.append({"cls": cls, "xyxy": [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]})
    return out


def load_cityscapes(split: str, variants=("clean", "0.005", "0.01", "0.02"), cities: list | None = None) -> dict[str, list[FrameGT]]:
    """One FrameGT list per variant, with IDENTICAL scenes in identical order (paired design)."""
    out = {v: [] for v in variants}
    gt_dir = CS_GT / split
    for city in sorted(os.listdir(gt_dir)):
        if cities and city not in cities:
            continue
        for fn in sorted(os.listdir(gt_dir / city)):
            m = _CS_RE.match(fn)
            if not m:
                continue
            stem = f"{m['city']}_{m['seq']}_{m['frame']}"
            boxes = cityscapes_boxes(gt_dir / city / fn)
            for v in variants:
                if v == "clean":
                    p = CS_CLEAN / split / city / f"{stem}_leftImg8bit.png"
                    stratum, weather, extra = "cs_clear", "clear", {}
                else:
                    p = CS_FOGGY / split / city / f"{stem}_leftImg8bit_foggy_beta_{v}.png"
                    stratum, weather, extra = f"cs_fog_{FOG_LEVELS[v]}", "fog", {"beta": v, "fog_level": FOG_LEVELS[v]}
                if not p.exists():
                    continue
                out[v].append(FrameGT(f"{stem}_{v}", str(p), "cityscapes", split, weather, "daytime", stratum, boxes,
                                      dict(extra, scene=stem, city=city)))
    return out


def build_strata(caps: dict, seed: int = 0, root: Path = PROJECT_ROOT) -> dict[str, list[FrameGT]]:
    """Fixed, seeded frame lists per stratum. caps: stratum -> max frames (None = all)."""
    rng = np.random.default_rng(seed)

    def take(frames: list[FrameGT], n: int | None) -> list[FrameGT]:
        frames = sorted(frames, key=lambda f: f.frame_id)
        if n is None or len(frames) <= n:
            return frames
        idx = sorted(rng.choice(len(frames), size=n, replace=False).tolist())
        return [frames[i] for i in idx]

    strata: dict[str, list[FrameGT]] = {}
    index = _bdd_image_index(root)
    print(f"[strata] BDD image index: {len(index)} files")
    train = load_bdd(BDD_LABELS / "bdd100k_labels_images_train.json", index, "train")
    val = load_bdd(BDD_LABELS / "bdd100k_labels_images_val.json", index, "val")
    strata["bdd_train_clear"] = take([f for f in train if f.weather_gt == "clear"], caps.get("bdd_train_clear", 3000))
    for cond in ("clear", "night", "rain", "snow"):
        strata[f"bdd_val_{cond}"] = take([f for f in val if f.weather_gt == cond], caps.get(f"bdd_val_{cond}"))
    for cond in ("fog", "night", "rain", "snow"):
        fr = load_acdc(cond, "train") + load_acdc(cond, "val")
        strata[f"acdc_{cond}"] = take(fr, caps.get(f"acdc_{cond}"))
    # Cityscapes: pick the scenes once, then the same scenes for every variant
    cs = {}
    for split in ("train", "val"):
        part = load_cityscapes(split)
        for v, frames in part.items():
            cs.setdefault(v, []).extend(frames)
    scenes = sorted({f.extra["scene"] for f in cs["clean"]})
    n = caps.get("cs_scenes")
    if n is not None and len(scenes) > n:
        scenes = sorted(rng.choice(scenes, size=n, replace=False).tolist())
    keep = set(scenes)
    for v, frames in cs.items():
        name = "cs_clear" if v == "clean" else f"cs_fog_{FOG_LEVELS[v]}"
        strata[name] = sorted([f for f in frames if f.extra["scene"] in keep], key=lambda f: f.extra["scene"])
    for k, v in strata.items():
        print(f"[strata] {k:18s} {len(v):6d} frames | boxes {sum(len(f.boxes) for f in v):7d} | persons "
              f"{sum(1 for f in v for b in f.boxes if b['cls'] == 'person'):6d}")
    return strata


def save_strata(strata: dict[str, list[FrameGT]], path: Path = RESULTS / "strata.json") -> None:
    save_json({k: [f.to_dict() for f in v] for k, v in strata.items()}, path)


def load_strata(path: Path = RESULTS / "strata.json") -> dict[str, list[FrameGT]]:
    d = json.load(open(path, encoding="utf-8"))
    return {k: [FrameGT(**f) for f in v] for k, v in d.items()}
