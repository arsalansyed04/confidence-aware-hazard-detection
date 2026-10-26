"""Weather classifier v2: ACDC + BDD100K + Foggy Cityscapes + RainCityscapes (separate experiment).

This does NOT touch the v1 model or its results (results/weather_classifier). It reuses the verified v1
building blocks from src/ (model, cache, transforms, metrics) and writes everything to a new folder.

What is different from v1
  data     Foggy Cityscapes (label fog) and RainCityscapes (label rain) join the TRAINING data. If the clean
           Cityscapes RGB package is extracted to <root>/leftImg8bit_trainvaltest/leftImg8bit it is picked up
           automatically as label clear (that removes the camera shortcut described in the README).
  splits   Cityscapes-family images are split BY CITY, shared by every Cityscapes source, so all fog levels and all
           36 rain renderings of one street scene always live in the same split:
               test = frankfurt, lindau, munster   (Cityscapes' official validation cities)
               val  = erfurt, weimar
               train= the remaining 16 cities
           ACDC and BDD100K keep exactly the v1 splits.
  sampling one epoch = --samples-per-epoch random draws. Every class gets 1/5 of the draws; inside a class each
           source's share is proportional to sqrt(number of scenes); inside a source every SCENE is equally likely;
           inside a scene every rendering is equally likely. Plain cross-entropy (label smoothing 0.05), because the
           sampler already balances the classes.
  memory   the number of DataLoader processes is capped from the RAM and Windows commit that are free at start.

Usage (Anaconda PowerShell Prompt, conda env "weather", project root as the working directory):
    python src_cityscapes/train_weather_cityscapes.py prepare     # manifest + split tables + cache of the rain images
    python src_cityscapes/train_weather_cityscapes.py train       # trains, then evaluates best.pt automatically
    python src_cityscapes/train_weather_cityscapes.py eval        # re-run the evaluation and the v1-vs-v2 comparison
Predict with the v2 model through the v1 script (same checkpoint format):
    python src/train_weather_classifier.py predict --out results/weather_classifier_cityscapes --predict-dir bdd100k-10k/val
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # must precede numpy/torch; every DataLoader worker inherits it (RAM)

import argparse
import ctypes
import gc
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import weather_data as wd  # noqa: E402  (v1 helper, reused unchanged)
import train_weather_classifier as v1  # noqa: E402  (v1 script, reused unchanged as a library)

CLASSES = wd.CLASSES
CS_TEST_CITIES = ("frankfurt", "lindau", "munster")  # Cityscapes' official validation cities
CS_VAL_CITIES = ("erfurt", "weimar")
CS_SOURCES = ("foggy_cs", "rain_cs", "clean_cs")
RAIN_DIR = Path("leftImg8bit_trainval_rain") / "leftImg8bit_rain"
CLEAN_DIRS = (Path("leftImg8bit_trainvaltest") / "leftImg8bit", Path("leftImg8bit_trainval") / "leftImg8bit",
              Path("leftImg8bit"))  # Cityscapes' own package name first; its unlabelled test/ folder is not used
_RAIN_RE = re.compile(
    r"^(?P<city>[a-z]+)_(?P<seq>\d+)_(?P<frame>\d+)_leftImg8bit_rain_alpha_(?P<alpha>[0-9.]+)_beta_(?P<beta>[0-9.]+)"
    r"_dropsize_(?P<drop>[0-9.]+)_pattern_(?P<pattern>\d+)\.png$")
_CLEAN_RE = re.compile(r"^(?P<city>[a-z]+)_(?P<seq>\d+)_(?P<frame>\d+)_leftImg8bit\.png$")

# memory model (measured on this laptop: a DataLoader worker that re-imports torch holds about 0.6 GB)
MAIN_PROCESS_GB, WORKER_GB, RESERVE_GB = 3.5, 0.75, 1.5


# --------------------------------------------------------------------------- CLI
def parse_args(argv=None) -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", type=Path, default=PROJECT_ROOT, help="project root")
    common.add_argument("--out", type=Path, default=Path("results/weather_classifier_cityscapes"),
                        help="output folder of THIS experiment; relative paths are taken relative to --root")
    common.add_argument("--cache-root", type=Path, default=Path("results/weather_classifier"),
                        help="folder whose cache/ subfolder holds the resized JPEGs; shared with v1 so the ~21k "
                             "images v1 already cached are not rebuilt (only cache files are written there)")
    common.add_argument("--v1-ckpt", type=Path, default=Path("results/weather_classifier/best.pt"),
                        help="v1 checkpoint evaluated on the same test rows for the comparison table")
    common.add_argument("--extra-ckpt", type=Path, default=Path("results/weather_classifier_cityscapes/best_no_clean.pt"),
                        help="optional third comparison column: the v2 model trained WITHOUT clean Cityscapes frames")
    common.add_argument("--img-size", type=int, default=256)
    common.add_argument("--cache-size", type=int, default=320)
    common.add_argument("--cache-workers", type=int, default=12, help="THREADS used to build the cache")
    common.add_argument("--workers", type=int, default=4, help="training DataLoader processes (auto-capped by free memory)")
    common.add_argument("--eval-workers", type=int, default=2, help="val/eval DataLoader processes (auto-capped)")
    common.add_argument("--eval-batch", type=int, default=256)
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--limit-per-class", type=int, default=None, metavar="N",
                        help="smoke test: keep at most N rows per (split, label)")
    common.add_argument("--bdd-labels", type=str, default=None)
    common.add_argument("--cpu", action="store_true")

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("prepare", parents=[common], help="manifest, split tables, cache")
    pp.add_argument("--manifest-only", action="store_true", help="write the manifest and tables but build no cache")
    t = sub.add_parser("train", parents=[common], help="train, then evaluate best.pt")
    t.add_argument("--epochs", type=int, default=15)
    t.add_argument("--samples-per-epoch", type=int, default=12000)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--no-final-eval", action="store_true")
    e = sub.add_parser("eval", parents=[common], help="evaluate best.pt (+ v1 on the same rows)")
    e.add_argument("--ckpt", type=Path, default=None)

    args = p.parse_args(argv)
    args.root = args.root.resolve()
    for name in ("out", "cache_root", "v1_ckpt", "extra_ckpt"):
        v = getattr(args, name)
        setattr(args, name, (v if v.is_absolute() else args.root / v).resolve())
    return args


# --------------------------------------------------------------------------- memory safety
def free_memory_gb() -> tuple[float | None, float | None]:
    """(free RAM, free Windows commit) in GB. Commit = RAM + page file; it is what actually ran out on 18 Sept."""
    ram = commit = None
    try:
        import psutil
        ram = psutil.virtual_memory().available / 2**30
    except Exception:  # noqa: BLE001
        pass
    if os.name == "nt":
        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                (k, ctypes.c_ulonglong) for k in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile",
                                                  "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual",
                                                  "ullAvailExtendedVirtual")]
        ms = _MS()
        ms.dwLength = ctypes.sizeof(_MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
            commit = ms.ullAvailPageFile / 2**30
    return ram, commit


def cap_workers(train_w: int, val_w: int, tag: str) -> tuple[int, int]:
    ram, commit = free_memory_gb()
    known = [x for x in (ram, commit) if x is not None]
    if not known:
        return train_w, val_w
    allowed = max(0, int((min(known) - MAIN_PROCESS_GB - RESERVE_GB) / WORKER_GB))
    new_val = min(val_w, max(0, allowed // 3))
    new_train = min(train_w, max(0, allowed - new_val))
    print(f"[memory] {tag}: free RAM {ram if ram is None else round(ram, 1)} GB, free commit "
          f"{commit if commit is None else round(commit, 1)} GB -> at most {allowed} loader processes; using "
          f"{new_train} train + {new_val} val/eval (requested {train_w} + {val_w})")
    if allowed < 2:
        print("[memory] very little memory is free: close other programs (browser!) for a faster run; "
              "continuing safely with in-process data loading")
    return new_train, new_val


# --------------------------------------------------------------------------- manifest
def cs_split(city: str) -> str:
    return "test" if city in CS_TEST_CITIES else "val" if city in CS_VAL_CITIES else "train"


def _walk_cityscapes(base: Path):
    for cs_folder in ("train", "val"):
        d = base / cs_folder
        if d.is_dir():
            for city in sorted(os.listdir(d)):
                if (d / city).is_dir():
                    for fn in sorted(os.listdir(d / city)):
                        yield d / city / fn, fn


def scan_foggy(root: Path) -> list[tuple]:
    rows = []
    for path, fn in _walk_cityscapes(root / wd.FOGGY_DIR):
        m = wd._FOGGY_RE.match(fn)
        if m:  # extra stays the plain beta string so the v1 CLIP script can sort it numerically
            rows.append(("foggy_cs", cs_split(m["city"]), "fog", m["city"], f"{m['seq']}_{m['frame']}", m["beta"], path))
    return rows


def scan_rain(root: Path) -> list[tuple]:
    rows = []
    for path, fn in _walk_cityscapes(root / RAIN_DIR):
        m = _RAIN_RE.match(fn)
        if m:
            extra = f"alpha_{m['alpha']}_beta_{m['beta']}_dropsize_{m['drop']}"  # the 3 rain intensities
            rows.append(("rain_cs", cs_split(m["city"]), "rain", m["city"], f"{m['seq']}_{m['frame']}", extra, path))
    return rows


def scan_clean(root: Path) -> list[tuple]:
    for cand in CLEAN_DIRS:
        if (root / cand).is_dir():
            rows = []
            for path, fn in _walk_cityscapes(root / cand):
                m = _CLEAN_RE.match(fn)
                if m:
                    rows.append(("clean_cs", cs_split(m["city"]), "clear", m["city"], f"{m['seq']}_{m['frame']}", "", path))
            print(f"[manifest] clean Cityscapes found at {root / cand}: {len(rows)} clear images (camera shortcut removed)")
            return rows
    print("[manifest] clean Cityscapes RGB not on disk: every Cityscapes-camera image is fog or rain "
          "(see README: the model can use the camera as a shortcut)")
    return []


def build_manifest(args) -> pd.DataFrame:
    root = args.root
    rows = (wd._scan_acdc(root) + wd._scan_bdd(root, args.bdd_labels)
            + scan_foggy(root) + scan_rain(root) + scan_clean(root))
    recs = []
    for source, split, label, group, frame, extra, path in rows:
        rel = Path(path).resolve().relative_to(root).as_posix()
        recs.append({"path": Path(path).resolve().as_posix(), "source": source, "split": split, "label": label,
                     "group": str(group), "frame": str(frame), "extra": str(extra),
                     "cache_path": wd.cache_path_for(args.cache_root, source, rel).as_posix()})
    df = pd.DataFrame.from_records(recs, columns=wd.MANIFEST_COLUMNS)
    return df.drop_duplicates("path").reset_index(drop=True)


def scene_key(df: pd.DataFrame) -> pd.Series:
    # the label is part of the key: an ACDC clear reference frame carries the same sequence and frame number as
    # its adverse-weather twin, and they are two different training scenes, not two renderings of one scene
    return df["source"] + "|" + df["group"] + "|" + df["frame"] + "|" + df["label"]


def check_splits(df: pd.DataFrame) -> dict:
    assert not df["path"].duplicated().any(), "leakage guard: duplicate image paths"
    ks = df.groupby(["source", "group", "frame"])["split"].nunique()
    assert (ks <= 1).all(), f"leakage guard: scenes in more than one split, e.g. {ks[ks > 1].index[:5].tolist()}"
    cs = df[df["source"].isin(CS_SOURCES)]
    per_city = cs.groupby("group")["split"].nunique()
    assert (per_city == 1).all(), f"leakage guard: Cityscapes cities in more than one split: {per_city[per_city > 1].index.tolist()}"
    cities = {s: sorted(cs.loc[cs["split"] == s, "group"].unique()) for s in ("train", "val", "test")}
    print("[leakage] duplicate paths: 0 | scenes in more than one split: 0 | Cityscapes cities in more than one split: 0")
    for s, c in cities.items():
        print(f"[splits] Cityscapes {s:5s} cities ({len(c)}): {', '.join(c)}")
    print("[splits] ACDC and BDD100K keep the v1 splits (ACDC official splits as shipped; labelled BDD100K -> train)")
    return {"cityscapes_cities": cities}


def tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    images = wd.counts_table(df)
    scenes = wd.counts_table(df.drop_duplicates(["source", "group", "frame", "split", "label"]))
    return images, scenes


def prepare_manifest(args) -> pd.DataFrame:
    args.out.mkdir(parents=True, exist_ok=True)
    df = build_manifest(args)
    info = check_splits(df)
    images, scenes = tables(df)
    print("[manifest] IMAGES per (source, split, label):\n" + images.to_string())
    print("[manifest] distinct SCENES per (source, split, label) (one scene = one street frame, any rendering):\n"
          + scenes.to_string())
    images.to_csv(args.out / "counts_images.csv")
    scenes.to_csv(args.out / "counts_scenes.csv")
    df.to_csv(args.out / "manifest.csv", index=False)
    info.update({"rows": int(len(df)), "sources": sorted(df["source"].unique().tolist()),
                 "test_cities": CS_TEST_CITIES, "val_cities": CS_VAL_CITIES,
                 "bdd_signature": wd.bdd_signature(args.root, args.bdd_labels)})
    wd.save_json(info, args.out / "manifest_info.json")
    print(f"[manifest] wrote {args.out / 'manifest.csv'} ({len(df)} rows)")
    return df


def get_manifest(args) -> pd.DataFrame:
    mp = args.out / "manifest.csv"
    if not mp.exists():
        print("[manifest] not found - building it first")
        return prepare_manifest(args)
    df = wd.load_manifest(mp)
    print(f"[manifest] loaded {mp} ({len(df)} rows)")
    check_splits(df)
    return df


def ensure_cache(df: pd.DataFrame, args) -> None:
    wd.ensure_cache(df["path"].tolist(), df["cache_path"].tolist(), args.cache_root, args.cache_size, args.cache_workers)


# --------------------------------------------------------------------------- sampling
def sampling_weights(train_df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Per-row draw probabilities: class 1/K, source share ~ sqrt(#scenes), scene uniform, rendering uniform."""
    scene = scene_key(train_df)
    t = pd.DataFrame({"label": train_df["label"].to_numpy(), "source": train_df["source"].to_numpy(),
                      "unit": (1.0 / scene.map(scene.value_counts())).to_numpy()})  # renderings of a scene sum to 1
    n_scenes = t.groupby(["label", "source"])["unit"].transform("sum")
    t["root"] = np.sqrt(n_scenes)
    class_total = t.drop_duplicates(["label", "source"]).groupby("label")["root"].sum()
    w = t["unit"] / n_scenes * (t["root"] / t["label"].map(class_total))
    w = w / w.sum()
    table = (t.assign(w=w).groupby(["label", "source"])
              .agg(images=("unit", "size"), scenes=("unit", "sum"), draw_share=("w", "sum")))
    table["scenes"] = table["scenes"].round().astype(int)
    return w.to_numpy(dtype=np.float64), table


# --------------------------------------------------------------------------- evaluation helpers
def predict_rows(ckpt: Path, rows: pd.DataFrame, args, device, workers: int, desc: str):
    model, meta = v1.load_model(ckpt, device)
    y = v1.labels_of(rows)
    ds = wd.CachedDataset(rows["cache_path"], y, meta["transform"])
    probs = v1.predict_probs(model, v1.make_loader(ds, args.eval_batch, False, workers), meta["norm"], meta["use_amp"], desc)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return y, probs, meta


def fraction_table(rows: pd.DataFrame, pred: np.ndarray, source: str, target: str, probs: np.ndarray | None = None) -> dict:
    """Per severity level (fog beta / rain intensity from the filename): how often the target class is predicted
    and, when probs are given, the mean target probability (does confidence rise with severity?)."""
    sub = rows["source"].to_numpy() == source
    out = {}
    for level in sorted(set(rows.loc[sub, "extra"])):
        m = sub & (rows["extra"].to_numpy() == level)
        p = pred[m]
        out[level] = {"n": int(m.sum()), f"fraction_pred_{target}": float((p == wd.CLASS_TO_IDX[target]).mean()),
                      "pred_distribution": {c: int((p == i).sum()) for i, c in enumerate(CLASSES)}}
        if probs is not None:
            out[level][f"mean_p_{target}"] = float(probs[m, wd.CLASS_TO_IDX[target]].mean())
    return out


def headline(rows: pd.DataFrame, y: np.ndarray, pred: np.ndarray) -> dict:
    """The numbers used in the v1-vs-v2 comparison, all on TEST rows."""
    src = rows["source"].to_numpy()
    h = {}
    m = src == "acdc"
    if m.any():
        cm = wd.classification_metrics(y[m], pred[m])
        h["acdc_test_accuracy"], h["acdc_test_macro_f1"] = cm["accuracy"], cm["macro_f1"]
    for level, v in fraction_table(rows, pred, "foggy_cs", "fog").items():
        h[f"foggy_cs_test_beta_{level}_called_fog"] = v["fraction_pred_fog"]
    for level, v in fraction_table(rows, pred, "rain_cs", "rain").items():
        h[f"rain_cs_test_{level}_called_rain"] = v["fraction_pred_rain"]
    m = src == "clean_cs"
    if m.any():
        h["clean_cs_test_called_clear"] = float((pred[m] == wd.CLASS_TO_IDX["clear"]).mean())
    return h


def plot_curves(history: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    a.plot(history["epoch"], history["train_loss"], color="#2f6fb3", linewidth=2, marker="o", markersize=4)
    a.set_title("Train loss (CE, label smoothing 0.05, balanced sampler)", fontsize=10)
    b.plot(history["epoch"], history["val_macro_f1"], color="#2f6fb3", linewidth=2, marker="o", markersize=4, label="val macro-F1")
    b.plot(history["epoch"], history["val_acc"], color="#c7702a", linewidth=2, marker="s", markersize=4, label="val accuracy")
    b.set_title("Validation: ACDC val + Cityscapes val cities", fontsize=10)
    b.set_ylim(0.0, 1.02)
    b.legend(frameon=False)
    for ax in (a, b):
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- commands
def cmd_prepare(args) -> None:
    df = prepare_manifest(args)
    if args.manifest_only:
        print("[prepare] --manifest-only: no cache built")
        return
    sub = wd.apply_limit(df, args.limit_per_class, args.seed)
    ensure_cache(sub, args)
    print(f"[prepare] done - cache under {args.cache_root / 'cache'}")


def cmd_train(args) -> None:
    device = v1.setup_device(args.cpu)
    use_amp = device.type == "cuda"
    v1.set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    print("[config] " + json.dumps(vars(args), default=str))
    df = wd.apply_limit(get_manifest(args), args.limit_per_class, args.seed)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    if train_df.empty or val_df.empty:
        sys.exit("[train] empty train or val set")
    ensure_cache(pd.concat([train_df, val_df]), args)

    weights, share = sampling_weights(train_df)
    print("[sampler] expected share of each epoch's draws (class 1/5, source ~ sqrt(scenes), scene uniform):\n"
          + share.to_string(float_format=lambda x: f"{x:.3f}"))
    share.to_csv(args.out / "sampling_shares.csv")

    train_w, val_w = cap_workers(args.workers, args.eval_workers, "train")
    y_train, y_val = v1.labels_of(train_df), v1.labels_of(val_df)
    train_ds = wd.CachedDataset(train_df["cache_path"], y_train,
                                wd.TrainTransform(args.img_size, scale=v1.RRC_SCALE, ratio=v1.RRC_RATIO))
    val_ds = wd.CachedDataset(val_df["cache_path"], y_val, wd.EvalTransform(args.img_size))
    gen = torch.Generator().manual_seed(args.seed)
    n_draws = min(args.samples_per_epoch, max(args.batch, len(train_ds) * 4))
    sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=n_draws,
                                    replacement=True, generator=gen)
    kw = dict(batch_size=args.batch, sampler=sampler, num_workers=train_w, pin_memory=True, drop_last=n_draws >= 2 * args.batch)
    if train_w > 0:
        kw.update(persistent_workers=True, prefetch_factor=4, worker_init_fn=wd.worker_init)
    train_loader = DataLoader(train_ds, **kw)
    val_loader = v1.make_loader(val_ds, args.eval_batch, False, val_w)
    print(f"[data] train pool {len(train_df)} images, {n_draws} draws per epoch | val {len(val_df)} images "
          f"{val_df['source'].value_counts().to_dict()}")
    print(f"[aug] RandomResizedCrop({args.img_size}, scale={v1.RRC_SCALE}, ratio={v1.RRC_RATIO}) + horizontal flip only; "
          "no photometric ops")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    print("[loss] CrossEntropyLoss(label_smoothing=0.05), no class weights: the sampler balances the classes")
    model = v1.build_model(pretrained=True).to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(v1.param_groups(model, args.lr, v1.WEIGHT_DECAY, v1.BACKBONE_LR_MULT),
                                  lr=args.lr, fused=(device.type == "cuda"))
    steps = len(train_loader)
    warmup, total = steps, args.epochs * steps
    norm = v1.InputNorm(wd.IMAGENET_MEAN, wd.IMAGENET_STD, device)
    print(f"[opt] AdamW(wd={v1.WEIGHT_DECAY}) head lr {args.lr:g}, backbone lr {args.lr * v1.BACKBONE_LR_MULT:g} | "
          f"{steps} steps/epoch, cosine over {total} steps, 1 epoch warm-up | bf16 autocast={use_amp}")

    best_f1, step, history, times = -1.0, 0, [], []
    best_path, t_start = args.out / "best.pt", time.perf_counter()
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        loss, step = v1.train_one_epoch(model, train_loader, criterion, optimizer, norm, use_amp, args.lr,
                                        warmup, total, step, f"epoch {epoch + 1}/{args.epochs}")
        probs = v1.predict_probs(model, val_loader, norm, use_amp, "val")
        m = wd.classification_metrics(y_val, probs.argmax(1))
        secs = time.perf_counter() - t0
        times.append(secs)
        is_best = m["macro_f1"] > best_f1
        history.append({"epoch": epoch + 1, "train_loss": loss, "val_acc": m["accuracy"], "val_macro_f1": m["macro_f1"],
                        "epoch_seconds": secs, "is_best": is_best})
        pd.DataFrame(history).to_csv(args.out / "history.csv", index=False)
        print(f"[epoch {epoch + 1:02d}/{args.epochs}] loss {loss:.4f} | val acc {m['accuracy']:.4f} | val macro-F1 "
              f"{m['macro_f1']:.4f} | {secs:.1f}s | ETA {np.mean(times) * (args.epochs - epoch - 1) / 60:.1f} min"
              + (" | new best" if is_best else ""))
        ckpt = v1.make_ckpt(model, args, epoch, {"val": m, "train_loss": loss})
        ckpt["experiment"] = "v2_cityscapes"
        if is_best:
            best_f1 = m["macro_f1"]
            torch.save(ckpt, best_path)
        torch.save(ckpt, args.out / "last.pt")
    plot_curves(pd.DataFrame(history), args.out / "training_curves.png")
    best = max(history, key=lambda h: h["val_macro_f1"])
    print(f"[train] done in {(time.perf_counter() - t_start) / 60:.1f} min | best epoch {best['epoch']} "
          f"val macro-F1 {best['val_macro_f1']:.4f} | {best_path}")

    del train_loader, val_loader, model, optimizer  # stop the loader processes before eval starts its own
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not args.no_final_eval:
        args.ckpt = best_path
        cmd_eval(args, device)


def cmd_eval(args, device=None) -> None:
    device = device or v1.setup_device(args.cpu)
    df = wd.apply_limit(get_manifest(args), args.limit_per_class, args.seed)
    ev = df[df["split"].isin(["val", "test"])].reset_index(drop=True)
    ensure_cache(ev, args)
    ckpt = getattr(args, "ckpt", None) or args.out / "best.pt"
    if not Path(ckpt).exists():
        sys.exit(f"[eval] checkpoint not found: {ckpt}")
    _, workers = cap_workers(0, args.eval_workers, "eval")
    y, probs, meta = predict_rows(Path(ckpt), ev, args, device, workers, "eval v2")
    pred = probs.argmax(1)
    out_rows = ev.copy()
    out_rows["pred"] = [CLASSES[i] for i in pred]
    for i, c in enumerate(CLASSES):
        out_rows[f"p_{c}"] = probs[:, i]
    out_rows.to_csv(args.out / "eval_predictions.csv", index=False)

    metrics = {"checkpoint": str(ckpt), "epoch": meta["epoch"], "classes": CLASSES, "per_source": {},
               "note": "foggy_cs / rain_cs test rows are unseen CITIES of a camera and renderer seen in training "
                       "(in-domain); only ACDC test is comparable with v1's headline accuracy"}
    for split in ("val", "test"):
        sm = (ev["split"] == split).to_numpy()
        metrics[split] = wd.classification_metrics(y[sm], pred[sm])
        metrics["per_source"][split] = {}
        for src in sorted(ev.loc[sm, "source"].unique()):
            m2 = sm & (ev["source"] == src).to_numpy()
            r = wd.classification_metrics(y[m2], pred[m2])
            metrics["per_source"][split][src] = r
            print(f"[eval] {split}/{src}: n={r['n']} acc={r['accuracy']:.4f} macro-F1={r['macro_f1']:.4f}")
    test = (ev["split"] == "test").to_numpy()
    trows, ty, tpred = ev[test].reset_index(drop=True), y[test], pred[test]
    tprobs = probs[test]
    metrics["foggy_cs_test_per_beta"] = fraction_table(trows, tpred, "foggy_cs", "fog", tprobs)
    metrics["rain_cs_test_per_intensity"] = fraction_table(trows, tpred, "rain_cs", "rain", tprobs)
    for name, mask in (("all", np.ones(len(trows), bool)), ("acdc", (trows["source"] == "acdc").to_numpy()),
                       ("cityscapes", trows["source"].isin(CS_SOURCES).to_numpy())):
        if mask.any():
            r = wd.classification_metrics(ty[mask], tpred[mask])
            wd.plot_confusion_matrix(r["confusion_matrix"], CLASSES,
                                     f"v2 test/{name}: n={r['n']}  acc={r['accuracy']:.3f}  macro-F1={r['macro_f1']:.3f}",
                                     args.out / f"confusion_matrix_test_{name}.png")

    comparison = {"v2": headline(trows, ty, tpred)}
    for column, other in (("v1", args.v1_ckpt), ("v2_no_clean", args.extra_ckpt)):
        if other.exists() and Path(ckpt).resolve() != other.resolve():
            yo, po, _ = predict_rows(other, trows, args, device, workers, f"eval {column}")
            comparison[column] = headline(trows, yo, po.argmax(1))
        else:
            print(f"[eval] no separate checkpoint for column {column} at {other}; column skipped")
    metrics["comparison_on_identical_test_rows"] = comparison
    table = pd.DataFrame(comparison)
    table = table[[c for c in ("v1", "v2_no_clean", "v2") if c in table.columns]]
    table.to_csv(args.out / "comparison_v1_vs_v2.csv", float_format="%.4f")
    print("[eval] SAME test rows | v1 = ACDC+BDD only | v2_no_clean = + Cityscapes fog and rain | v2 = this checkpoint:\n"
          + table.to_string(float_format=lambda x: f"{x:.4f}"))
    wd.save_json(metrics, args.out / "metrics.json")
    print(f"[eval] wrote metrics.json, comparison_v1_vs_v2.csv, confusion_matrix_test_*.png, eval_predictions.csv -> {args.out}")


def main(argv=None) -> None:
    args = parse_args(argv)
    wd.start_log(args.out / f"log_{args.cmd}.txt")
    {"prepare": cmd_prepare, "train": cmd_train, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
