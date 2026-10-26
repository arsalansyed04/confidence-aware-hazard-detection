"""Library for weather classifier v4: data, severity labels, depth-based weather rendering, model.

v4 predicts three things per image with one backbone:
    condition   clear | fog | night | rain | snow
    fog level   none | light | medium | dense      (visibility about >1 km | 600 m | 300 m | 150 m)
    rain level  none | light | medium | heavy

Severity labels exist only where they are known:
    * every clear-weather image of every camera  -> fog none, rain none
    * Foggy Cityscapes beta 0.005 / 0.01 / 0.02   -> fog light / medium / dense (and rain none)
    * RainCityscapes intensity 1 / 2 / 3          -> rain light / medium / heavy
    * fog and rain RENDERED on the fly onto clear images of all three cameras with an estimated depth map
      (atmospheric scattering model, continuous beta) -> the level used for rendering
    * real fog / rain / snow / night images        -> unknown, masked out of the severity losses
Rendering is what stops the severity heads from being tied to the Cityscapes camera.

Reuses the verified v1 / v2 code (src/, src_cityscapes/) as libraries; nothing there is modified.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # plain HTTP: the Xet backend stalled on this network

import json
import math
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
for _p in (PROJECT_ROOT / "src_cityscapes", PROJECT_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import weather_data as wd  # noqa: E402
import train_weather_classifier as v1  # noqa: E402
import train_weather_cityscapes as v2  # noqa: E402

CLASSES = wd.CLASSES
FOG_LEVELS = ["none", "light", "medium", "dense"]
RAIN_LEVELS = ["none", "light", "medium", "heavy"]
DBF_BETA_TO_LEVEL = {"0.005": 1, "0.01": 2, "0.02": 3}
RAIN_ALPHA_TO_LEVEL = {"0.01": 1, "0.02": 2, "0.03": 3}
# level bins in beta: edges are the geometric means between the DBF betas, so 0.005 / 0.01 / 0.02 sit mid-bin
BETA_MIN, BETA_LIGHT_MED, BETA_MED_DENSE, BETA_MAX = 0.004, 0.005 * 2**0.5, 0.01 * 2**0.5, 0.02 * 2**0.5
EVAL_BETAS = ("0.005", "0.01", "0.02")
RAIN_HAZE_BETA = (0.005, 0.010, 0.015)  # RainCityscapes couples haze to rain intensity the same way
TRAIN_DOMAINS = ("acdc", "bdd", "cityscapes")
RENDER_BOOST = 0.35  # rendered sources get a smaller share of a class's draws than real ones
V3_COLUMNS = wd.MANIFEST_COLUMNS + ["domain", "fog_level", "rain_level", "render", "depth_path", "in_v2_train"]
DAWN_PREFIX = {"foggy": "fog", "haze": "fog", "mist": "fog", "rain_storm": "rain", "snow_storm": "snow"}
DEFAULT_FOG_PARAMS = {"d_near": 6.0, "q0": 0.012}

CONFIGS = {  # name -> recipe; LODO picks one, `train --config` overrides
    "mnv3_full": dict(backbone="mobilenet_v3_large", tune="full", lr=3e-4, backbone_lr_mult=0.3, img=256, batch=64),
    "convnext_full": dict(backbone="convnext_tiny", tune="full", lr=2e-4, backbone_lr_mult=0.1, img=256, batch=32),
    "dinov2_frozen": dict(backbone="dinov2_vits14", tune="frozen", lr=1e-3, backbone_lr_mult=0.0, img=252, batch=64),
    "dinov2_last2": dict(backbone="dinov2_vits14", tune="last2", lr=5e-4, backbone_lr_mult=0.04, img=252, batch=48),
    "dinov2_last4": dict(backbone="dinov2_vits14", tune="last4", lr=5e-4, backbone_lr_mult=0.04, img=252, batch=48),
}
FINALISTS = ["dinov2_last2", "dinov2_last4", "convnext_full"]  # the first LODO put mnv3_full and dinov2_frozen clearly behind


def stable_int(text: str) -> int:
    return zlib.crc32(text.encode("utf-8"))


def beta_to_level(beta: float) -> int:
    return 1 if beta < BETA_LIGHT_MED else 2 if beta < BETA_MED_DENSE else 3


# --------------------------------------------------------------------------- manifest
def levels_for(source: str, label: str, extra: str) -> tuple[int, int]:
    """(fog_level, rain_level); -1 = unknown, masked out of that head's loss."""
    fog = rain = -1
    if label == "clear":
        fog, rain = 0, 0
    if source == "foggy_cs":
        fog, rain = DBF_BETA_TO_LEVEL[extra], 0
    if source == "rain_cs":
        rain = RAIN_ALPHA_TO_LEVEL[extra.split("_")[1]]
    return fog, rain


def find_bdd100k_images(root: Path) -> Path | None:
    """Folder that holds train/ and val/ of the FULL BDD100K image set (Kaggle solesensei layout or similar)."""
    skip = {"results", "src", "src_cityscapes", "src_v3", "bdd100k-10k", "rgb_anon_trainvaltest_ACDC",
            "gt_detection_trainval", "gtFine_trainvaltest", "leftImg8bit_trainvaltest",
            "leftImg8bit_trainval_foggyDBF", "leftImg8bit_trainval_rain"}
    stack = [(root, 0)]
    while stack:
        d, depth = stack.pop()
        try:
            subs = [x for x in os.scandir(d) if x.is_dir()]
        except OSError:
            continue
        for s in subs:
            if depth == 0 and s.name in skip:
                continue
            if s.name == "100k" and (Path(s.path) / "train").is_dir() and (Path(s.path) / "val").is_dir():
                return Path(s.path)
            if depth < 5:
                stack.append((Path(s.path), depth + 1))
    return None


def scan_bdd_v3(root: Path, bdd_labels: str | None, images_100k: Path | None, cap: int, seed: int) -> list[tuple]:
    """BDD100K rows. Split by LABEL FILE: the official val labels are the held-out TEST set; 5% of the training
    videos (by video id) are the validation set. Works with the 10k subset alone or with the full 100k images."""
    files = wd.find_bdd_label_files(root, bdd_labels)
    if not files:
        print("[manifest] no BDD label JSON found; BDD skipped")
        return []
    index, small_train = {}, set()
    for split in ("train", "val", "test"):
        d = root / wd.BDD_DIR / split
        if d.is_dir():
            for fn in os.listdir(d):
                index[fn] = d / fn
                if split == "train":
                    small_train.add(fn)
    big = images_100k or find_bdd100k_images(root)
    if big:
        n0 = len(index)
        for split in ("train", "val"):
            # recursive on purpose: the Kaggle package nests the 70,000 training images in trainA/, trainB/, testA/, testB/
            for dirpath, _, names in os.walk(big / split):
                for fn in names:
                    if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                        index[fn] = Path(dirpath) / fn  # prefer the full-set copy
        print(f"[manifest] full BDD100K images found at {big} ({len(index) - n0} new files)")
    else:
        print("[manifest] full BDD100K images NOT on disk: only the 10k subset is used (see README for the download)")
    rows = []
    for f in files:
        held_out = "val" in f.stem.lower()
        print(f"[manifest] reading {f.name} ({f.stat().st_size / 1e6:.0f} MB) ...")
        for rec in wd._iter_json_records(f):
            name = Path(str(rec.get("name", ""))).name
            label = wd.bdd_weather_label(rec.get("attributes") or {})
            if not name or label is None or name not in index:
                continue
            stem = Path(name).stem
            video = stem.split("-")[0]
            split = "test" if held_out else ("val" if stable_int(video) % 20 == 0 else "train")
            rows.append(("bdd", split, label, video, stem, "", index[name], int(name in small_train)))
    rng = np.random.default_rng(seed)
    keep = []
    by = {}
    for i, r in enumerate(rows):
        by.setdefault((r[1], r[2]), []).append(i)
    for (split, label), idx in by.items():
        if split == "train" and len(idx) > cap:
            idx = sorted(rng.choice(idx, size=cap, replace=False).tolist())
            print(f"[manifest] BDD train/{label}: capped to {cap} images")
        keep += idx
    return [rows[i] for i in sorted(keep)]


def scan_dawn(root: Path) -> list[tuple]:
    """DAWN (real fog / rain / snow from unknown cameras): TEST ONLY, never trained on. Sandstorm images skipped."""
    for cand in ("DAWN", "dawn", "dawn-dataset", "dawn_dataset", "DAWN_dataset"):
        d = root / cand
        if d.is_dir():
            rows, seen = [], set()
            for p in wd.list_images(d):
                stem = p.stem.lower()
                label = next((v for k, v in DAWN_PREFIX.items() if stem.startswith(k)), None)
                if label is None:
                    label = next((x for x in (part.lower() for part in p.parts) if x in ("fog", "rain", "snow")), None)
                if label and stem not in seen:
                    seen.add(stem)
                    rows.append(("dawn", "test", label, "dawn", p.stem, "", p, 0))
            print(f"[manifest] DAWN found at {d}: {len(rows)} fog/rain/snow images (external test only)")
            return rows
    print("[manifest] DAWN not on disk (optional external test; see README)")
    return []


def dawn_sand_images(root: Path) -> list[Path]:
    """DAWN sandstorm / dust images: a weather the model is never taught. Used only as an out-of-distribution probe."""
    for cand in ("DAWN", "dawn", "dawn-dataset", "dawn_dataset", "DAWN_dataset"):
        d = root / cand
        if d.is_dir():
            seen, out = set(), []
            for p in wd.list_images(d):
                s = p.stem.lower()
                if s.startswith(("sand_storm", "dusttornado")) and s not in seen:
                    seen.add(s)
                    out.append(p)
            return out
    return []


def build_manifest(args) -> pd.DataFrame:
    root = args.root
    recs = []

    def add(source, split, label, group, frame, extra, path, domain, in_v2=0):
        p = Path(path).resolve()
        try:
            rel = p.relative_to(root).as_posix()
        except ValueError:
            rel = p.as_posix()
        fog, rain = levels_for(source, label, str(extra))
        recs.append({"path": p.as_posix(), "source": source, "split": split, "label": label, "group": str(group),
                     "frame": str(frame), "extra": str(extra),
                     "cache_path": wd.cache_path_for(args.cache_root, source, rel).as_posix(), "domain": domain,
                     "fog_level": fog, "rain_level": rain, "render": "", "depth_path": "", "in_v2_train": int(in_v2)})

    for r in wd._scan_acdc(root):
        add(*r, domain="acdc", in_v2=int(r[1] == "train"))
    for r in scan_bdd_v3(root, args.bdd_labels, args.bdd100k_images, args.bdd_cap, args.seed):
        add(*r[:7], domain="bdd", in_v2=r[7])
    for r in v2.scan_foggy(root) + v2.scan_rain(root) + v2.scan_clean(root):
        add(*r, domain="cityscapes", in_v2=int(r[1] == "train"))
    for r in scan_dawn(root):
        add(*r[:7], domain="dawn")
    df = pd.DataFrame.from_records(recs, columns=V3_COLUMNS).drop_duplicates("path").reset_index(drop=True)
    return pd.concat([df, virtual_rows(df, args)], ignore_index=True)


def virtual_rows(df: pd.DataFrame, args) -> pd.DataFrame:
    """Rows that render fog / rain onto a clear image. Train rows draw a new severity on every access; val/test rows
    have a fixed severity derived from the file path, so evaluation is reproducible."""
    clear = df[(df["label"] == "clear") & df["domain"].isin(TRAIN_DOMAINS)]
    parts = []
    for (source, split), g in clear.groupby(["source", "split"], sort=True):
        cap = args.render_train_cap if split == "train" else args.render_eval_cap
        if len(g) > cap:
            g = g.sample(cap, random_state=args.seed)
        depth = [(args.cache_root / "cache" / "depth" / source / (Path(c).stem + ".png")).as_posix() for c in g["cache_path"]]
        for kind in ("fog", "rain"):
            v = g.copy()
            v["source"], v["label"], v["render"], v["depth_path"] = f"{source}_r{kind}", kind, kind, depth
            pick = np.array([stable_int(p + kind) % 3 for p in v["path"]])
            if kind == "fog":
                v["rain_level"] = 0
                v["fog_level"] = pick + 1 if split != "train" else -1
                v["extra"] = [EVAL_BETAS[i] for i in pick] if split != "train" else ""
            else:
                v["fog_level"] = -1
                v["rain_level"] = pick + 1 if split != "train" else -1
                v["extra"] = [f"level_{i + 1}" for i in pick] if split != "train" else ""
            parts.append(v)
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in V3_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} lacks columns {missing}; run prepare again")
    for c in ("fog_level", "rain_level", "in_v2_train"):
        df[c] = df[c].astype(int)
    return df


def check_manifest(df: pd.DataFrame) -> None:
    real = df[df["render"] == ""]
    assert not real["path"].duplicated().any(), "leakage guard: duplicate image paths"
    ks = df.groupby(["source", "group", "frame"])["split"].nunique()
    assert (ks <= 1).all(), f"leakage guard: a scene appears in two splits: {ks[ks > 1].index[:5].tolist()}"
    cs = real[real["domain"] == "cityscapes"]
    per_city = cs.groupby("group")["split"].nunique()
    assert (per_city == 1).all(), "leakage guard: a Cityscapes city appears in two splits"
    bv = real[real["domain"] == "bdd"].groupby("group")["split"].nunique()
    n_shared = int((bv > 1).sum())
    assert not ((real["domain"] == "dawn") & (real["split"] != "test")).any(), "DAWN must be test-only"
    base = real.set_index("path")["split"]
    virt = df[df["render"] != ""]
    assert (virt["split"].to_numpy() == base.loc[virt["path"]].to_numpy()).all(), "a rendered row left its base split"
    print(f"[leakage] duplicate paths 0 | scenes in two splits 0 | Cityscapes cities in two splits 0 | rendered rows "
          f"follow their base split | BDD videos spanning train and test labels: {n_shared} (official BDD split, kept)")


def counts(df: pd.DataFrame) -> pd.DataFrame:
    return wd.counts_table(df)


# --------------------------------------------------------------------------- depth maps
def prior_inverse_depth(h: int, w: int) -> np.ndarray:
    """Fallback when no depth network is available: flat ground plane, horizon at 47% of the height, far above it."""
    y = np.arange(h, dtype=np.float32)[:, None]
    yh = 0.47 * h
    q = np.clip((y - yh) / (h - yh), 0.0, 1.0)
    return cv2.GaussianBlur(np.repeat(q, w, axis=1), (0, 0), 3)


def load_depth_model(name: str, device: torch.device):
    """Returns infer(list of BGR uint8, same shape) -> list of inverse-depth float arrays, or None for the prior."""
    if name == "prior":
        print("[depth] using the geometric ground-plane prior (no network): cruder fog, but needs no download")
        return None
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError:
        sys.exit("[depth] the 'transformers' package is missing. Run:  pip install transformers\n"
                 "        (or pass --depth-model prior to use the network-free fallback)")
    print(f"[depth] loading {name} (first run downloads about 100 MB from Hugging Face over plain HTTP)")
    proc = AutoImageProcessor.from_pretrained(name)
    model = AutoModelForDepthEstimation.from_pretrained(name).to(device).eval()

    def infer(batch_bgr):
        rgb = [cv2.cvtColor(b, cv2.COLOR_BGR2RGB) for b in batch_bgr]
        inputs = proc(images=rgb, return_tensors="pt").to(device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(**inputs).predicted_depth
        return [x.float().cpu().numpy() for x in out]

    return infer


def _save_depth(q: np.ndarray, shape_hw: tuple[int, int], dst: str) -> None:
    lo, hi = np.percentile(q, 1), np.percentile(q, 99)
    q = np.clip((q - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    h, w = shape_hw
    q = cv2.resize(q.astype(np.float32), (max(w // 2, 8), max(h // 2, 8)), interpolation=cv2.INTER_AREA)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".tmp.png"
    cv2.imwrite(tmp, np.round(q * 65535).astype(np.uint16))
    os.replace(tmp, dst)


def compute_depth_maps(df: pd.DataFrame, device: torch.device, model_name: str, batch: int = 12) -> None:
    from tqdm import tqdm

    need = df[df["render"] != ""].drop_duplicates("depth_path")
    todo = [(c, d) for c, d in zip(need["cache_path"], need["depth_path"]) if not os.path.exists(d)]
    print(f"[depth] {len(need) - len(todo)} of {len(need)} depth maps already exist; computing {len(todo)}")
    if not todo:
        return
    infer = load_depth_model(model_name, device)
    buf_img, buf_dst, shape = [], [], None

    def flush():
        if not buf_img:
            return
        maps = infer(buf_img) if infer else [prior_inverse_depth(*buf_img[0].shape[:2]) for _ in buf_img]
        for m, im, dst in zip(maps, buf_img, buf_dst):
            _save_depth(m, im.shape[:2], dst)
        buf_img.clear()
        buf_dst.clear()

    for cache_path, dst in tqdm(todo, desc="depth", unit="img", mininterval=1.0):
        im = wd.imread(cache_path)
        if im is None:
            raise RuntimeError(f"cannot read {cache_path}; run prepare first")
        if shape is not None and im.shape != shape or len(buf_img) >= batch:
            flush()
        shape = im.shape
        buf_img.append(im)
        buf_dst.append(dst)
    flush()


def load_depth(path: str, shape_hw: tuple[int, int]) -> np.ndarray:
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise RuntimeError(f"missing depth map {path}; run the depth command")
    q = d.astype(np.float32) / 65535.0
    return cv2.resize(q, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_LINEAR)


# --------------------------------------------------------------------------- weather rendering
def estimate_airlight(bgr: np.ndarray) -> np.ndarray:
    """Atmospheric light (He et al.): mean colour of the 0.1% brightest dark-channel pixels, kept bright and greyish."""
    dark = bgr.min(axis=2).ravel()
    n = max(1, dark.size // 1000)
    idx = np.argpartition(dark, -n)[-n:]
    a = bgr.reshape(-1, 3)[idx].astype(np.float32).mean(axis=0)
    a = 0.5 * a + 0.5 * a.mean()
    return np.clip(a, 170.0, 255.0)


def render_fog(bgr: np.ndarray, q: np.ndarray, beta: float, params: dict, airlight: np.ndarray | None = None) -> np.ndarray:
    """I = J*t + A*(1-t), t = exp(-beta*d), metric depth d = d_near / (q + q0)**gamma from normalised inverse depth q."""
    d = params["d_near"] / np.power(q + params["q0"], params.get("gamma", 1.0))
    t = np.exp(-beta * d).astype(np.float32)[..., None]
    a = estimate_airlight(bgr) if airlight is None else airlight
    out = bgr.astype(np.float32) * t + a * (1.0 - t)
    return np.clip(out, 0, 255).astype(np.uint8)


def _streak_kernel(length: int, angle: float) -> np.ndarray:
    k = np.zeros((length, length), np.float32)
    k[:, length // 2] = 1.0
    m = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    k = cv2.warpAffine(k, m, (length, length))
    return k / max(float(k.sum()), 1e-6)


def render_rain(bgr: np.ndarray, q: np.ndarray, level: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    """Rain = depth-dependent haze (as in RainCityscapes) + motion-blurred streaks; density and length grow with level."""
    out = render_fog(bgr, q, RAIN_HAZE_BETA[level - 1] * rng.uniform(0.85, 1.15), params).astype(np.float32)
    h, w = out.shape[:2]
    density = (0.006, 0.012, 0.020)[level - 1] * rng.uniform(0.8, 1.2)
    length = max(5, int((9, 13, 17)[level - 1] * h / 320) | 1)
    mask = (rng.random((h, w)) < density).astype(np.float32)
    streaks = np.clip(cv2.filter2D(mask, -1, _streak_kernel(length, rng.uniform(-25, 25))) * length, 0, 1)  # fine 1-px streaks
    alpha = (streaks * rng.uniform(0.22, 0.5))[..., None]
    return np.clip(out * (1 - alpha) + 225.0 * alpha, 0, 255).astype(np.uint8)


def calibration_pairs(df: pd.DataFrame, n_pairs: int, seed: int) -> pd.DataFrame:
    """Clean Cityscapes training scenes that have a depth map AND the real DBF fog frame at all three betas."""
    pairs = df[(df["source"] == "clean_cs_rfog") & (df["split"] == "train")][["group", "frame", "cache_path", "depth_path"]]
    for beta in EVAL_BETAS:
        dbf = df[(df["source"] == "foggy_cs") & (df["extra"] == beta)][["group", "frame", "path", "cache_path"]]
        pairs = pairs.merge(dbf.rename(columns={"path": f"dbf_path_{beta}", "cache_path": f"dbf_cache_{beta}"}), on=["group", "frame"])
    pairs = pairs[[os.path.exists(p) for p in pairs["depth_path"]]]
    return pairs.sample(min(n_pairs, len(pairs)), random_state=seed) if len(pairs) else pairs


def calibrate_fog(pairs: pd.DataFrame) -> dict:
    """Fit (d_near, q0, gamma) so fog rendered on clean Cityscapes frames matches the REAL Foggy Cityscapes-DBF frame
    of the same scene at all three betas. The objective is pixel error plus a penalty on signed bias, because a
    renderer that is systematically too dense (the 19 Sept result: about one level) mislabels the severity it teaches."""
    data = []
    for r in pairs.to_dict("records"):
        j = wd.imread(r["cache_path"])
        tgt = [wd.imread(r[f"dbf_cache_{b}"]) for b in EVAL_BETAS]
        if j is None or any(t is None or t.shape != j.shape for t in tgt):
            continue
        h, w = j.shape[0] // 2, j.shape[1] // 2  # half resolution: 4x faster, same optimum
        small = lambda im: cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)  # noqa: E731
        data.append((small(j), [small(t).astype(np.float32) for t in tgt], load_depth(r["depth_path"], (h, w)), estimate_airlight(j)))
    if not data:
        print("[calibrate] no readable clean/DBF pairs with depth maps; keeping the default fog parameters")
        return dict(DEFAULT_FOG_PARAMS, calibrated=False)
    betas = [float(b) for b in EVAL_BETAS]

    def score(p):
        mae, bias = np.zeros(3), np.zeros(3)
        for j, tgt, q, a in data:
            for k, beta in enumerate(betas):
                diff = render_fog(j, q, beta, p, a).astype(np.float32) - tgt[k]
                mae[k] += np.abs(diff).mean()
                bias[k] += diff.mean()
        return mae / len(data), bias / len(data)

    best = None
    for d_near in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0):
        for q0 in (0.01, 0.015, 0.02, 0.025, 0.035, 0.05, 0.07, 0.1):
            for gamma in (0.8, 1.0, 1.25):
                p = {"d_near": d_near, "q0": q0, "gamma": gamma}
                mae, bias = score(p)
                obj = float(mae.mean() + 0.5 * np.abs(bias).mean())
                if best is None or obj < best[0]:
                    best = (obj, p, mae, bias)
    nothing = np.mean([[np.abs(j.astype(np.float32) - t).mean() for t in tgt] for j, tgt, _, _ in data], axis=0)
    _, p, mae, bias = best
    out = dict(p, calibrated=True, pairs=len(data), mae_per_beta=dict(zip(EVAL_BETAS, map(float, mae))),
               signed_bias_per_beta=dict(zip(EVAL_BETAS, map(float, bias))), mae_doing_nothing_per_beta=dict(zip(EVAL_BETAS, map(float, nothing))))
    print(f"[calibrate] {len(data)} Cityscapes scenes x 3 fog levels -> d_near={p['d_near']}, q0={p['q0']}, gamma={p['gamma']}")
    for k, b in enumerate(EVAL_BETAS):
        print(f"[calibrate]   beta {b}: pixel error {mae[k]:.2f} (doing nothing {nothing[k]:.2f}) | signed bias {bias[k]:+.2f} "
              f"(positive = my fog is brighter/denser than the real DBF fog)")
    return out


# --------------------------------------------------------------------------- dataset
def camera_aug(bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Breaks camera fingerprints (colour cast, resolution, JPEG history) without touching brightness or contrast.
    Applied AFTER weather rendering so a rendered image is not recognisable by its missing compression artefacts."""
    h, w = bgr.shape[:2]
    if rng.random() < 0.3:
        bgr = np.clip(bgr.astype(np.float32) * rng.uniform(0.93, 1.07, size=3), 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        s = rng.uniform(0.5, 0.9)
        small = cv2.resize(bgr, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
        bgr = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.4:
        ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35, 91))])
        if ok:
            bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return bgr


class V3Dataset:
    """Returns (uint8 RGB image, condition, fog_level, rain_level, row index). Picklable for Windows workers."""

    def __init__(self, df: pd.DataFrame, img_size: int, train: bool, fog_params: dict, camera_augment: bool = True):
        self.cache = df["cache_path"].tolist()
        self.depth = df["depth_path"].tolist()
        self.render = df["render"].tolist()
        self.extra = df["extra"].tolist()
        self.cond = df["label"].map(wd.CLASS_TO_IDX).astype(int).tolist()
        self.fog = df["fog_level"].astype(int).tolist()
        self.rain = df["rain_level"].astype(int).tolist()
        self.train, self.params, self.cam = train, dict(fog_params), camera_augment
        self.tf = wd.TrainTransform(img_size, scale=v1.RRC_SCALE, ratio=v1.RRC_RATIO) if train else wd.EvalTransform(img_size)

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, i):
        bgr = wd.imread(self.cache[i])
        if bgr is None:
            raise RuntimeError(f"cannot read {self.cache[i]}; run prepare")
        fog, rain, kind = self.fog[i], self.rain[i], self.render[i]
        rng = np.random.default_rng() if self.train else np.random.default_rng(stable_int(self.cache[i] + kind))
        if kind:
            q = load_depth(self.depth[i], bgr.shape[:2])
            # evaluation is reproducible: the stored severity is used, or (for a training row that is being evaluated
            # as part of a held-out camera) one derived from the file path
            pick = stable_int(self.cache[i] + kind) % 3
            if kind == "fog":
                if self.train:
                    beta = math.exp(rng.uniform(math.log(BETA_MIN), math.log(BETA_MAX)))
                else:
                    beta = float(self.extra[i]) if self.extra[i] else float(EVAL_BETAS[pick])
                bgr, fog = render_fog(bgr, q, beta, self.params), beta_to_level(beta)
            else:
                level = int(rng.integers(1, 4)) if self.train else (rain if rain > 0 else pick + 1)
                bgr, rain = render_rain(bgr, q, level, self.params, rng), level
        if self.train and self.cam:
            bgr = camera_aug(bgr, rng)
        return self.tf(bgr), self.cond[i], fog, rain, i


def sampling_weights(train_df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Class 1/K; inside a class a source's share ~ sqrt(scenes) (x RENDER_BOOST for rendered sources);
    inside a source every scene equally likely; inside a scene every rendering equally likely."""
    scene = train_df["source"] + "|" + train_df["group"] + "|" + train_df["frame"] + "|" + train_df["label"]
    t = pd.DataFrame({"label": train_df["label"].to_numpy(), "source": train_df["source"].to_numpy(),
                      "unit": (1.0 / scene.map(scene.value_counts())).to_numpy()})
    n = t.groupby(["label", "source"])["unit"].transform("sum")
    boost = np.where(t["source"].str.contains("_rfog|_rrain", regex=True), RENDER_BOOST, 1.0)
    t["root"] = np.sqrt(n) * boost
    total = t.drop_duplicates(["label", "source"]).groupby("label")["root"].sum()
    w = t["unit"] / n * (t["root"] / t["label"].map(total))
    w = w / w.sum()
    table = t.assign(w=w).groupby(["label", "source"]).agg(images=("unit", "size"), scenes=("unit", "sum"), draw_share=("w", "sum"))
    table["scenes"] = table["scenes"].round().astype(int)
    return w.to_numpy(dtype=np.float64), table


# --------------------------------------------------------------------------- model
class _DinoFeatures(nn.Module):
    def __init__(self, pretrained: bool, img_size: int):
        super().__init__()
        import timm
        self.vit = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=pretrained, num_classes=0, img_size=img_size)
        self.out_dim = self.vit.embed_dim * 2

    def forward(self, x):
        t = self.vit.forward_features(x)  # (B, 1 + patches, C)
        k = self.vit.num_prefix_tokens
        return torch.cat([t[:, 0], t[:, k:].mean(dim=1)], dim=1)  # class token + mean patch token (haze is global)


def make_backbone(name: str, pretrained: bool, img_size: int) -> tuple[nn.Module, int]:
    import torchvision.models as tvm
    if name == "mobilenet_v3_large":
        m = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
        m.classifier = nn.Sequential(m.classifier[0], m.classifier[1])  # keep the pretrained 960 -> 1280 layer
        return m, 1280
    if name == "convnext_tiny":
        m = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[2] = nn.Identity()
        return m, 768
    if name == "dinov2_vits14":
        m = _DinoFeatures(pretrained, img_size)
        return m, m.out_dim
    raise ValueError(f"unknown backbone {name}")


class WeatherNetV3(nn.Module):
    def __init__(self, backbone: str, tune: str, img_size: int, pretrained: bool = True):
        super().__init__()
        self.backbone_name, self.tune, self.img_size = backbone, tune, img_size
        self.backbone, feat = make_backbone(backbone, pretrained, img_size)
        self.trunk = nn.Sequential(nn.LayerNorm(feat), nn.Linear(feat, 512), nn.GELU(), nn.Dropout(0.2))
        self.cond, self.fog, self.rain = nn.Linear(512, len(CLASSES)), nn.Linear(512, 4), nn.Linear(512, 4)
        if tune == "frozen" or tune.startswith("last"):
            for p in self.backbone.parameters():
                p.requires_grad = False
        if tune.startswith("last"):  # "last2", "last4": unfreeze only the last N blocks
            n = int(tune[4:])
            if backbone == "dinov2_vits14":
                parts = list(self.backbone.vit.blocks[-n:]) + [self.backbone.vit.norm]
            else:
                parts = list(self.backbone.features[-n:])
            for part in parts:
                for p in part.parameters():
                    p.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self.tune == "frozen":
            self.backbone.eval()
        return self

    def forward(self, x):
        if self.tune == "frozen":
            with torch.no_grad():
                f = self.backbone(x)
        else:
            f = self.backbone(x)
        z = self.trunk(f.float())
        return self.cond(z), self.fog(z), self.rain(z)


def param_groups(model: WeatherNetV3, lr: float, backbone_mult: float, weight_decay: float = 0.05) -> list[dict]:
    groups = []
    for is_head, params in ((True, [p for n, p in model.named_parameters() if not n.startswith("backbone.") and p.requires_grad]),
                            (False, [p for n, p in model.named_parameters() if n.startswith("backbone.") and p.requires_grad])):
        for decay in (True, False):
            ps = [p for p in params if (p.ndim >= 2) == decay]
            if ps:
                groups.append({"params": ps, "lr": lr, "lr_mult": 1.0 if is_head else backbone_mult,
                               "weight_decay": weight_decay if decay else 0.0})
    return groups


def save_checkpoint(path: Path, model: WeatherNetV3, cfg: dict, cfg_name: str, fog_params: dict, extra: dict) -> None:
    ck = {"experiment": "weather_v4", "model": model.state_dict(), "config_name": cfg_name, "config": cfg,
          "classes": CLASSES, "fog_levels": FOG_LEVELS, "rain_levels": RAIN_LEVELS, "fog_params": fog_params,
          "mean": list(wd.IMAGENET_MEAN), "std": list(wd.IMAGENET_STD), **extra}
    tmp = str(path) + ".tmp"
    torch.save(ck, tmp)
    os.replace(tmp, path)


def load_model(ckpt_path, device="cuda"):
    """(model, meta) for the runtime pipeline; meta holds the eval transform and the GPU normaliser."""
    device = torch.device(device)
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = WeatherNetV3(cfg["backbone"], cfg["tune"], cfg["img"], pretrained=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    meta = {"config": cfg, "config_name": ck.get("config_name"), "device": device, "use_amp": device.type == "cuda",
            "transform": wd.EvalTransform(cfg["img"]), "norm": v1.InputNorm(ck["mean"], ck["std"], device),
            "metrics": ck.get("metrics"), "epoch": ck.get("epoch"), "fog_params": ck.get("fog_params", dict(DEFAULT_FOG_PARAMS))}
    return model, meta


def load_ensemble(ckpt_paths, device="cuda") -> list:
    """[(model, meta), ...] for predict_image_ensemble; pass every results/weather_v4/ckpt_*.pt."""
    return [load_model(p, device) for p in ckpt_paths]


def _probs(model, meta, bgr: np.ndarray, tta: bool):
    x = meta["norm"](torch.from_numpy(meta["transform"](bgr)[None]))
    with torch.inference_mode(), v1.autocast_ctx(meta["use_amp"]):
        outs = [model(x)] + ([model(x.flip(3))] if tta else [])  # flip(3): mirror left-right
    return [np.mean([torch.softmax(o[k].float(), 1)[0].cpu().numpy() for o in outs], axis=0) for k in range(3)]


def predict_image_ensemble(models: list, bgr_ndarray: np.ndarray, tta: bool = True) -> dict:
    """Average of all models (and of the mirrored image). Same output dictionary as predict_image."""
    per = [_probs(m, meta, bgr_ndarray, tta) for m, meta in models]
    pc, pf, pr = (np.mean([p[k] for p in per], axis=0) for k in range(3))
    weather = CLASSES[int(pc.argmax())]
    return {"weather": weather, "fog_level": level_name(pf, FOG_LEVELS, weather == "fog"),
            "rain_level": level_name(pr, RAIN_LEVELS, weather == "rain"),
            "p_weather": dict(zip(CLASSES, map(float, pc))), "p_fog_level": dict(zip(FOG_LEVELS, map(float, pf))),
            "p_rain_level": dict(zip(RAIN_LEVELS, map(float, pr)))}


def level_name(probs, names: list[str], active: bool) -> str:
    """Severity reported to the pipeline. 'none' when the weather is not that condition; 'unrated' when the weather
    IS that condition but the severity head itself answers 'none' (measured 19 Sept 2026: this happens on 99% of REAL
    rain, because rain severity could only be learned from synthetic rain). Never invents a level."""
    if not active:
        return "none"
    k = int(np.argmax(probs))
    return names[k] if k > 0 else "unrated"


def predict_image(model, meta, bgr_ndarray) -> dict:
    """{'weather', 'fog_level', 'rain_level', 'p_weather': {...}, 'p_fog_level': {...}, 'p_rain_level': {...}}.
    fog_level is reported only when the condition is fog, rain_level only when it is rain (else 'none')."""
    x = torch.from_numpy(meta["transform"](bgr_ndarray)[None])
    with torch.inference_mode(), v1.autocast_ctx(meta["use_amp"]):
        c, f, r = model(meta["norm"](x))
    pc, pf, pr = (torch.softmax(t.float(), 1)[0].cpu().numpy() for t in (c, f, r))
    weather = CLASSES[int(pc.argmax())]
    return {"weather": weather,
            "fog_level": level_name(pf, FOG_LEVELS, weather == "fog"),
            "rain_level": level_name(pr, RAIN_LEVELS, weather == "rain"),
            "p_weather": dict(zip(CLASSES, map(float, pc))), "p_fog_level": dict(zip(FOG_LEVELS, map(float, pf))),
            "p_rain_level": dict(zip(RAIN_LEVELS, map(float, pr)))}


def save_json(obj, path: Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
