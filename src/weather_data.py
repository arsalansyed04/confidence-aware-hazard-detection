"""Data utilities for the weather classifier (torch-free).

Shared by train_weather_classifier.py and clip_zeroshot_weather.py:

* manifest construction  - ACDC (adverse + clear *_ref frames, official splits kept
                           as shipped), Foggy Cityscapes (eval-only OOD probe) and
                           optional BDD100K (only when a labels JSON exists)
* leakage guards         - asserts on duplicate paths / cross-split keys / foggy rows
* resized JPEG cache     - every image decoded once, short side --cache-size, JPEG q92
* cv2-based transforms   - RandomResizedCrop+HFlip (train), Resize+CenterCrop (eval),
                           returning uint8 RGB HWC; normalisation happens on the GPU
* map-style datasets     - plain classes with __len__/__getitem__ (DataLoader-compatible)
* metrics / plots        - sklearn + matplotlib helpers used by both scripts

This module deliberately does not import torch: the cache-building worker
processes (spawned on Windows) re-import it and stay light.
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# --------------------------------------------------------------------------- constants
CLASSES = ["clear", "fog", "night", "rain", "snow"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MANIFEST_COLUMNS = ["path", "source", "split", "label", "group", "frame", "extra", "cache_path"]
REAL_SOURCES = ("acdc", "bdd")  # foggy_cs is a synthetic, eval-only probe
JPEG_QUALITY = 92
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

ACDC_DIR = Path("rgb_anon_trainvaltest_ACDC") / "rgb_anon"
ACDC_CONDITIONS = ("fog", "night", "rain", "snow")
ACDC_SPLITS = ("train", "val", "test")
FOGGY_DIR = Path("leftImg8bit_trainval_foggyDBF") / "leftImg8bit_foggyDBF"
FOGGY_REFINED = Path("leftImg8bit_trainval_foggyDBF") / "foggy_trainval_refined_filenames.txt"
BDD_DIR = Path("bdd100k-10k")
BDD_LABELS_GLOB = "bdd100k-10k/labels/*.json"

_ACDC_RE = re.compile(r"^(?P<seq>[^_]+)_frame_(?P<frame>\d+)_rgb(?P<ref>_ref)?_anon\.png$")
_FOGGY_RE = re.compile(
    r"^(?P<city>[a-z]+)_(?P<seq>\d+)_(?P<frame>\d+)_leftImg8bit_foggy_beta_(?P<beta>[0-9.]+)\.png$"
)
_BDD_WEATHER = {
    "clear": "clear", "overcast": "clear", "partly cloudy": "clear",
    "rainy": "rain", "foggy": "fog", "snowy": "snow",
}


# --------------------------------------------------------------------------- manifest
def _scan_acdc(root: Path) -> list[tuple]:
    base = root / ACDC_DIR
    if not base.is_dir():
        raise FileNotFoundError(f"ACDC folder not found: {base}")
    rows = []
    for cond in ACDC_CONDITIONS:
        for split in ACDC_SPLITS:
            # the condition folder is the weather label; every *_ref frame is clear weather
            for sub, label in ((split, cond), (f"{split}_ref", "clear")):
                d = base / cond / sub
                if not d.is_dir():
                    raise FileNotFoundError(f"missing ACDC folder: {d}")
                is_ref = sub.endswith("_ref")
                for seq in sorted(os.listdir(d)):
                    sd = d / seq
                    if not sd.is_dir():
                        continue
                    for fn in sorted(os.listdir(sd)):
                        m = _ACDC_RE.match(fn)
                        if m is None:
                            continue
                        if bool(m["ref"]) != is_ref:
                            raise RuntimeError(f"unexpected ACDC filename {fn} in {sd}")
                        rows.append(("acdc", split, label, m["seq"], m["frame"], "", sd / fn))
    return rows


def _scan_foggy(root: Path) -> list[tuple]:
    base = root / FOGGY_DIR
    if not base.is_dir():
        print(f"[manifest] Foggy Cityscapes not found at {base}; probe skipped")
        return []
    rows = []
    for cs_split in ("train", "val"):  # both Cityscapes splits become our eval-only 'test'
        d = base / cs_split
        if not d.is_dir():
            continue
        for city in sorted(os.listdir(d)):
            cd = d / city
            if not cd.is_dir():
                continue
            for fn in sorted(os.listdir(cd)):
                m = _FOGGY_RE.match(fn)
                if m is None:
                    continue
                rows.append(("foggy_cs", "test", "fog", m["city"], f"{m['seq']}_{m['frame']}", m["beta"], cd / fn))
    return rows


def bdd_weather_label(attrs: dict) -> str | None:
    """timeofday==night -> night; daytime -> weather mapping; dawn/dusk & undefined dropped."""
    tod = str(attrs.get("timeofday", "undefined")).lower()
    if tod == "night":
        return "night"
    if tod != "daytime":
        return None
    return _BDD_WEATHER.get(str(attrs.get("weather", "undefined")).lower())


def _iter_json_records(path: Path):
    """Yield the records of a BDD label file one at a time.

    The top-level array is scanned with JSONDecoder.raw_decode so a ~1 GB
    bdd100k_labels_images_train.json never becomes a single giant object tree.
    """
    text = path.read_text(encoding="utf-8")
    i, n = 0, len(text)
    while i < n and text[i].isspace():
        i += 1
    if i < n and text[i] == "[":
        dec = json.JSONDecoder()
        i += 1
        while True:
            while i < n and text[i] in " \t\r\n,":
                i += 1
            if i >= n or text[i] == "]":
                return
            obj, i = dec.raw_decode(text, i)
            yield obj
    else:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for key in ("frames", "images", "labels", "data"):
                if isinstance(obj.get(key), list):
                    obj = obj[key]
                    break
        if not isinstance(obj, list):
            raise ValueError(f"unrecognised BDD label file layout: {path}")
        yield from obj


def find_bdd_label_files(root: Path, spec: str | None) -> list[Path]:
    if spec:
        p = Path(spec)
        if p.is_file():
            return [p]
        pattern = spec if p.is_absolute() else str(root / spec)
    else:
        pattern = str(root / BDD_LABELS_GLOB)
    return sorted(Path(x) for x in glob.glob(pattern))


def _scan_bdd(root: Path, label_spec: str | None) -> list[tuple]:
    files = find_bdd_label_files(root, label_spec)
    if not files:
        print(f"[manifest] no BDD label JSON found ({label_spec or BDD_LABELS_GLOB}); BDD skipped")
        return []
    on_disk: dict[str, tuple[str, Path]] = {}
    for split in ("train", "val"):  # BDD train images -> train, val images -> val
        d = root / BDD_DIR / split
        if d.is_dir():
            for fn in os.listdir(d):
                on_disk[fn] = (split, d / fn)
    rows, n_rec, n_drop, n_missing = [], 0, 0, 0
    for f in files:
        print(f"[manifest] reading BDD labels {f} ({f.stat().st_size / 1e6:.1f} MB)")
        for rec in _iter_json_records(f):
            n_rec += 1
            name = rec.get("name")
            label = bdd_weather_label(rec.get("attributes") or {})
            if label is None or not name:
                n_drop += 1
                continue
            hit = on_disk.get(name) or on_disk.get(Path(name).name)
            if hit is None:
                n_missing += 1
                continue
            split, p = hit
            stem = Path(name).stem
            rows.append(("bdd", split, label, stem.split("-")[0], stem, "", p))
    print(f"[manifest] BDD: {n_rec} records, {n_drop} dropped (dawn/dusk/undefined), "
          f"{n_missing} not on disk, {len(rows)} used")
    return rows


def cache_path_for(out: Path, source: str, rel_path: str) -> Path:
    h = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:20]
    return out / "cache" / source / f"{h}.jpg"


def build_manifest(root: Path, out: Path, bdd_labels: str | None = None) -> pd.DataFrame:
    root, out = Path(root).resolve(), Path(out).resolve()
    rows = _scan_acdc(root) + _scan_foggy(root) + _scan_bdd(root, bdd_labels)
    if not rows:
        raise RuntimeError("no images found")
    recs = []
    for source, split, label, group, frame, extra, path in rows:
        rel = path.resolve().relative_to(root).as_posix()
        recs.append({
            "path": path.resolve().as_posix(), "source": source, "split": split, "label": label,
            "group": group, "frame": frame, "extra": extra,
            "cache_path": cache_path_for(out, source, rel).as_posix(),
        })
    df = pd.DataFrame.from_records(recs, columns=MANIFEST_COLUMNS)
    before = len(df)
    df = df.drop_duplicates("path").reset_index(drop=True)  # a label file listing an image twice is not a leak
    if len(df) != before:
        print(f"[manifest] dropped {before - len(df)} duplicate BDD records")
    return df


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} lacks columns {missing}; re-run prepare")
    return df


def check_leakage(df: pd.DataFrame) -> dict:
    """Hard asserts (duplicates, cross-split keys, foggy outside test) + printed ACDC sequence overlap."""
    dup = int(df["path"].duplicated().sum())
    assert dup == 0, f"leakage guard: {dup} duplicate image paths in the manifest"

    key_splits = df.groupby(["source", "group", "frame"])["split"].nunique()
    shared = key_splits[key_splits > 1]
    assert len(shared) == 0, (
        f"leakage guard: {len(shared)} (source, group, frame) keys appear in more than one split, e.g. "
        f"{shared.index[:5].tolist()}"
    )

    foggy_bad = int(((df["source"] == "foggy_cs") & (df["split"] != "test")).sum())
    assert foggy_bad == 0, f"leakage guard: {foggy_bad} foggy_cs rows outside the test split (eval-only set)"

    acdc = df[df["source"] == "acdc"]
    seq_splits = acdc.groupby("group")["split"].nunique()
    n_multi = int((seq_splits > 1).sum())
    print("[leakage] duplicate paths: 0 | cross-split (source,group,frame) keys: 0 | foggy_cs outside test: 0")
    print(f"[info] ACDC sequence ids appearing in more than one split: {n_multi} of {len(seq_splits)} "
          "(official ACDC splits used exactly as shipped; they are not sequence-disjoint, "
          "but every (sequence, frame) pair lives in one split only)")
    return {"acdc_sequences_total": int(len(seq_splits)), "acdc_sequences_in_multiple_splits": n_multi}


def counts_table(df: pd.DataFrame) -> pd.DataFrame:
    tab = (df.groupby(["source", "split", "label"]).size()
             .unstack("label", fill_value=0)
             .reindex(columns=CLASSES, fill_value=0))
    tab["total"] = tab.sum(axis=1)
    return tab


def apply_limit(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    """Smoke-test subset: keep at most n rows per (split, label), sampled with a fixed seed."""
    if not n:
        return df
    sub = df.sample(frac=1.0, random_state=seed).groupby(["split", "label"], sort=False).head(n)
    return sub.sort_index().reset_index(drop=True)


# --------------------------------------------------------------------------- image cache
def imread(path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    img = cv2.imread(path, flags)
    if img is None and os.path.exists(path):  # cv2.imread cannot open non-ASCII paths on Windows
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    return img


def imread_reduced(path: str, min_short_side: int) -> np.ndarray | None:
    """Decode at half resolution (libjpeg DCT-domain scaling for JPEG) unless that would
    leave the short side below `min_short_side`, in which case decode at full size."""
    img = imread(path, cv2.IMREAD_REDUCED_COLOR_2)
    if img is not None and min(img.shape[:2]) < min_short_side:
        img = imread(path, cv2.IMREAD_COLOR)
    return img


def _cache_one(src: str, dst: str, cache_size: int) -> str | None:
    """Resize one source image to `cache_size` short side and save as JPEG q92. Returns an error string or None."""
    try:
        if os.path.exists(dst):
            return None
        img = imread_reduced(src, cache_size)
        if img is None:
            return f"unreadable: {src}"
        h, w = img.shape[:2]
        scale = cache_size / min(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        tmp = dst + ".tmp.jpg"
        if not cv2.imwrite(tmp, img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
            return f"write failed: {dst}"
        os.replace(tmp, dst)  # atomic: a killed run never leaves a truncated file that passes the exists() check
        return None
    except Exception as e:  # noqa: BLE001 - reported by the parent
        return f"{type(e).__name__}: {e} ({src})"


def ensure_cache(paths, cache_paths, out: Path, cache_size: int, workers: int) -> None:
    """Build every missing cache file for the given rows (skips files that already exist).

    A THREAD pool is used on purpose: cv2 releases the GIL while decoding, resizing and encoding,
    so threads scale across cores, and no worker processes are spawned (on Windows every spawned
    process re-imports the main script, torch included, which can exhaust RAM)."""
    out = Path(out)
    info_path = out / "cache" / "cache_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        if int(info.get("cache_size", cache_size)) != cache_size:
            raise RuntimeError(
                f"existing cache at {out / 'cache'} was built with --cache-size {info['cache_size']}, "
                f"requested {cache_size}; delete the cache folder or pass the same --cache-size"
            )
    todo = [(s, d) for s, d in zip(paths, cache_paths) if not os.path.exists(d)]
    print(f"[cache] {len(paths) - len(todo)} of {len(paths)} images already cached; building {len(todo)}")
    if not todo:
        return
    for d in {os.path.dirname(d) for _, d in todo}:
        os.makedirs(d, exist_ok=True)
    if not info_path.exists():
        info_path.parent.mkdir(parents=True, exist_ok=True)
        info_path.write_text(json.dumps({"cache_size": cache_size, "jpeg_quality": JPEG_QUALITY}, indent=2))
    srcs, dsts = zip(*todo)
    workers = max(1, workers)
    cv2.setNumThreads(1)  # one decode per thread; the pool provides the parallelism
    with ThreadPoolExecutor(max_workers=workers) as ex:
        it = ex.map(_cache_one, srcs, dsts, repeat(cache_size))
        errors = [e for e in tqdm(it, total=len(todo), desc=f"cache x{workers} threads", unit="img") if e]
    if errors:
        raise RuntimeError(f"{len(errors)} images failed to cache, first: {errors[:3]}")


# --------------------------------------------------------------------------- transforms (uint8, cv2)
def _rrc_params(h: int, w: int, scale: tuple, ratio: tuple) -> tuple[int, int, int, int]:
    """Same sampler as torchvision RandomResizedCrop.get_params (10 tries, then centre-crop fallback)."""
    area = h * w
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
    for _ in range(10):
        target = area * random.uniform(scale[0], scale[1])
        aspect = math.exp(random.uniform(log_ratio[0], log_ratio[1]))
        cw = int(round(math.sqrt(target * aspect)))
        ch = int(round(math.sqrt(target / aspect)))
        if 0 < cw <= w and 0 < ch <= h:
            return random.randint(0, h - ch), random.randint(0, w - cw), ch, cw
    in_ratio = w / h
    if in_ratio < ratio[0]:
        cw, ch = w, int(round(w / ratio[0]))
    elif in_ratio > ratio[1]:
        ch, cw = h, int(round(h * ratio[1]))
    else:
        cw, ch = w, h
    return (h - ch) // 2, (w - cw) // 2, ch, cw


def _resize(img: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    # INTER_AREA is antialiased for downscaling (like PIL); INTER_LINEAR for upscaling
    interp = cv2.INTER_AREA if size_wh[0] < img.shape[1] else cv2.INTER_LINEAR
    return cv2.resize(img, size_wh, interpolation=interp)


class TrainTransform:
    """RandomResizedCrop(img_size, scale, ratio) + RandomHorizontalFlip, nothing else.

    Input BGR uint8 HWC (cv2), output RGB uint8 HWC. No photometric ops: luminance
    statistics are the night/fog cue. Uses the `random` module, which the DataLoader
    seeds per worker. Picklable (plain class, no lambdas)."""

    def __init__(self, img_size: int, scale=(0.6, 1.0), ratio=(0.9, 1.1)):
        self.size, self.scale, self.ratio = int(img_size), tuple(scale), tuple(ratio)

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        top, left, ch, cw = _rrc_params(h, w, self.scale, self.ratio)
        out = _resize(bgr[top:top + ch, left:left + cw], (self.size, self.size))
        if random.random() < 0.5:
            out = out[:, ::-1]
        return np.ascontiguousarray(out[:, :, ::-1])  # BGR -> RGB


class EvalTransform:
    """Resize(img_size) (short side, aspect kept) + CenterCrop(img_size); BGR uint8 in, RGB uint8 out."""

    def __init__(self, img_size: int):
        self.size = int(img_size)

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        s = self.size
        if h <= w:
            nh, nw = s, int(s * w / h)
        else:
            nh, nw = int(s * h / w), s
        if (nh, nw) != (h, w):
            bgr = _resize(bgr, (nw, nh))
        top, left = (nh - s) // 2, (nw - s) // 2
        return np.ascontiguousarray(bgr[top:top + s, left:left + s, ::-1])


# --------------------------------------------------------------------------- datasets (DataLoader-compatible)
def _read_or_raise(path: str, reduced_min_side: int | None) -> np.ndarray:
    img = imread_reduced(path, reduced_min_side) if reduced_min_side else imread(path)
    if img is None:
        raise RuntimeError(f"cannot read image {path} (run prepare to rebuild the cache?)")
    return img


class CachedDataset:
    """(uint8 RGB HWC image, label index) from the resized cache. Default collate stacks to (B,H,W,3) uint8."""

    def __init__(self, cache_paths, labels, transform):
        self.paths = [str(p) for p in cache_paths]
        self.labels = [int(x) for x in labels]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(_read_or_raise(self.paths[i], None)), self.labels[i]


class ImageFileDataset:
    """(uint8 RGB HWC image, index) for arbitrary image files (predict). Large sources decode at half res."""

    def __init__(self, paths, transform, min_short_side: int):
        self.paths = [str(p) for p in paths]
        self.transform = transform
        self.min_short_side = int(min_short_side)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(_read_or_raise(self.paths[i], self.min_short_side)), i


def worker_init(_worker_id: int) -> None:
    cv2.setNumThreads(0)  # DataLoader workers are already one process each


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())


# --------------------------------------------------------------------------- metrics & plots
def classification_metrics(y_true, y_pred, classes=CLASSES) -> dict:
    """accuracy, macro-F1 (over classes present in y_true/y_pred, sklearn default), per-class P/R/F1, confusion."""
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    labels = list(range(len(classes)))
    if len(y_true) == 0:
        return {"n": 0}
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, c in enumerate(classes)},
        "pred_distribution": {c: int((y_pred == i).sum()) for i, c in enumerate(classes)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def plot_confusion_matrix(cm, classes, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=np.int64)
    row = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    norm = cm / row
    fig, ax = plt.subplots(figsize=(6.6, 5.8), dpi=150)
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]}\n{100 * norm[i, j]:.1f}%", ha="center", va="center", fontsize=9,
                    color="white" if norm[i, j] > 0.55 else "#222222")
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row fraction")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    ep = history["epoch"]
    ax1.plot(ep, history["train_loss"], color="#2f6fb3", linewidth=2, marker="o", markersize=4)
    ax1.set_title("Train loss (class-weighted CE, ls=0.05)", fontsize=10)
    ax1.set_xlabel("epoch")
    ax2.plot(ep, history["val_macro_f1"], color="#2f6fb3", linewidth=2, marker="o", markersize=4, label="val macro-F1")
    ax2.plot(ep, history["val_acc"], color="#c7702a", linewidth=2, marker="s", markersize=4, label="val accuracy")
    ax2.set_title("ACDC val (model selection = macro-F1)", fontsize=10)
    ax2.set_xlabel("epoch")
    ax2.set_ylim(0.0, 1.02)
    ax2.legend(frameon=False)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_json(obj, path: Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- manifest staleness + logging
def bdd_signature(root: Path, spec: str | None) -> list[list]:
    """[[name, size], ...] of the BDD label files on disk; stored in manifest_info.json to detect a stale manifest."""
    return [[p.name, p.stat().st_size] for p in find_bdd_label_files(Path(root), spec)]


class _Tee:
    """Mirror stdout to a log file (tqdm bars go to stderr and are deliberately not mirrored)."""

    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)
        self.fh.flush()

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def start_log(path: Path) -> None:
    """Append everything printed from now on to `path` (one file per command, e.g. <out>/log_train.txt)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a", encoding="utf-8")
    fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} | {' '.join(sys.argv)}\n")
    sys.stdout = _Tee(sys.stdout, fh)
    print(f"[log] console output is also appended to {path}")
