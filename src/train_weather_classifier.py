"""Weather classifier (clear / fog / night / rain / snow) for the perception pipeline.

Runtime-only component: it is evaluated against dataset labels with a confusion
matrix and is never used to stratify detection results.

Model    torchvision mobilenet_v3_large (IMAGENET1K_V2), classifier[-1] -> Linear(1280, 5), all layers fine-tuned
Data     ACDC official train/val/test splits (adverse folders labelled by condition, every *_ref frame = clear),
         Foggy Cityscapes as an eval-only OOD probe (split=test, extra=beta), BDD100K only if a labels JSON exists
Speed    one-off resized JPEG cache (short side --cache-size), cv2 decode, uint8 batches normalised on the GPU,
         channels_last + bf16 autocast (no GradScaler), persistent pinned DataLoader workers, fused AdamW

Usage (Anaconda PowerShell Prompt, conda env "weather" active, project root as the working directory):
    python src/train_weather_classifier.py prepare                       # manifest + counts + resized cache (once)
    python src/train_weather_classifier.py train                         # 12 epochs -> best.pt / last.pt, then eval
    python src/train_weather_classifier.py eval                          # metrics.json, confusion matrices, foggy probe
    python src/train_weather_classifier.py predict --predict-dir bdd100k-10k/val
Smoke test first, with a separate --out so the real best.pt can never be overwritten:
    python src/train_weather_classifier.py train --out results/smoke --limit-per-class 32 --epochs 1 --workers 2
Every output lands under --out (default results/weather_classifier, relative to --root) and every command also
appends its console output to <out>/log_<command>.txt.

Importable for the runtime pipeline:
    from train_weather_classifier import load_model, predict_image
    model, meta = load_model("results/weather_classifier/best.pt", "cuda")
    label, probs = predict_image(model, meta, bgr_frame)   # bgr_frame: HxWx3 uint8 from cv2
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # must precede numpy/torch; workers inherit it (RAM, see README)
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import weather_data as wd  # noqa: E402  (helper next to this file)

CLASSES = wd.CLASSES
BACKBONE_LR_MULT = 0.3   # frozen recipe: backbone LR = 0.3 x head LR
WEIGHT_DECAY = 0.05      # frozen recipe: AdamW weight decay
RRC_SCALE = (0.6, 1.0)   # frozen recipe: RandomResizedCrop area range
RRC_RATIO = (0.9, 1.1)   # frozen recipe: RandomResizedCrop aspect range
DEFAULT_CACHE_WORKERS = max(2, min(16, (os.cpu_count() or 8) - 2))
CUDA_HELP = (
    "Install the CUDA 12.8 build inside your conda env:\n"
    "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128\n"
    "(the RTX 5060 is sm_120 and needs the cu128 wheels), or pass --cpu for a slow CPU run."
)


# --------------------------------------------------------------------------- CLI
def parse_args(argv=None) -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", type=Path, default=SCRIPT_DIR.parent, help="project root (default: parent of src/)")
    common.add_argument("--out", type=Path, default=Path("results/weather_classifier"),
                        help="output folder; a relative path is taken relative to --root")
    common.add_argument("--img-size", type=int, default=256)
    common.add_argument("--cache-size", type=int, default=320, help="short side of the resized JPEG cache")
    common.add_argument("--workers", type=int, default=6, help="DataLoader workers")
    common.add_argument("--cache-workers", type=int, default=DEFAULT_CACHE_WORKERS,
                        help="process-pool size for building the cache")
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--limit-per-class", type=int, default=None, metavar="N",
                        help="smoke test: keep at most N rows per (split, label)")
    common.add_argument("--bdd-labels", type=str, default=None,
                        help="BDD100K labels JSON or glob (default: auto-detect bdd100k-10k/labels/*.json)")
    common.add_argument("--cpu", action="store_true", help="run on CPU (no CUDA check)")
    common.add_argument("--eval-workers", type=int, default=4, help="DataLoader workers for eval/predict passes")
    common.add_argument("--eval-batch", type=int, default=256, help="batch size for val/test/predict passes")

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare", parents=[common], help="build manifest.csv, counts.csv and the resized cache")

    t = sub.add_parser("train", parents=[common], help="fine-tune (runs prepare first if the manifest is missing)")
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=3e-4, help=f"head LR; the backbone uses {BACKBONE_LR_MULT} x this")
    t.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    t.add_argument("--no-final-eval", action="store_true", help="skip the automatic eval of best.pt after training")

    e = sub.add_parser("eval", parents=[common], help="evaluate best.pt on val and test (+ foggy probe)")
    e.add_argument("--ckpt", type=Path, default=None, help="checkpoint (default <out>/best.pt)")

    pr = sub.add_parser("predict", parents=[common], help="predict a folder of unlabelled images to CSV")
    pr.add_argument("--ckpt", type=Path, default=None, help="checkpoint (default <out>/best.pt)")
    pr.add_argument("--predict-dir", type=Path, required=True, help="folder searched recursively for jpg/png")
    pr.add_argument("--limit", type=int, default=None, help="only the first N images (sorted)")

    args = p.parse_args(argv)
    args.root = args.root.resolve()
    args.out = (args.out if args.out.is_absolute() else args.root / args.out).resolve()
    if args.cmd == "predict":
        pdir = args.predict_dir
        if not pdir.is_absolute():  # relative paths are taken relative to --root first, then to the working directory
            pdir = (args.root / pdir) if (args.root / pdir).exists() else pdir.resolve()
        args.predict_dir = pdir
        if "leftImg8bit_trainval_rain" in pdir.as_posix():
            sys.exit("[predict] leftImg8bit_trainval_rain is excluded from this project and is never read")
    return args


# --------------------------------------------------------------------------- torch helpers
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_device(cpu: bool) -> torch.device:
    """Print versions, pick the device and run a 512x512 CUDA matmul kernel test."""
    print(f"[env] python {sys.version.split()[0]} | torch {torch.__version__} | torchvision {torchvision.__version__} "
          f"| CUDA build {torch.version.cuda}")
    if cpu:
        print("[env] --cpu given: running on the CPU")
        torch.set_num_threads(max(1, (os.cpu_count() or 8) - 2))
        return torch.device("cpu")
    if not torch.cuda.is_available():
        sys.exit(f"[env] CUDA is not available in this Python ({sys.executable}).\n{CUDA_HELP}")
    dev = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 2**30
    try:
        a = torch.randn(512, 512, device=dev)
        s = float((a @ a).sum())
        torch.cuda.synchronize()
        if not math.isfinite(s):
            raise RuntimeError("matmul returned a non-finite value")
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"[env] CUDA kernel test failed on {name} (sm_{cap[0]}{cap[1]}): {exc}\n{CUDA_HELP}")
    print(f"[env] GPU {name} | sm_{cap[0]}{cap[1]} | {mem:.1f} GB | 512x512 matmul kernel test OK")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return dev


def autocast_ctx(use_amp: bool):
    return torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()


def build_model(num_classes: int = len(CLASSES), pretrained: bool = True) -> nn.Module:
    from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)  # 1280 -> 5
    return model


def param_groups(model: nn.Module, lr: float, weight_decay: float, backbone_mult: float) -> list[dict]:
    """Head (new Linear) at lr, backbone at backbone_mult*lr; no weight decay on biases/BN params."""
    head_ids = {id(p) for p in model.classifier[-1].parameters()}
    buckets: dict[tuple[bool, bool], list] = {}
    for p in model.parameters():
        if p.requires_grad:
            buckets.setdefault((id(p) in head_ids, p.ndim >= 2), []).append(p)
    groups = []
    for (is_head, decay), params in buckets.items():
        groups.append({
            "params": params, "lr": lr, "lr_mult": 1.0 if is_head else backbone_mult,
            "weight_decay": weight_decay if decay else 0.0,
            "name": f"{'head' if is_head else 'backbone'}/{'decay' if decay else 'no_decay'}",
        })
    return groups


def lr_factor(step: int, warmup: int, total: int) -> float:
    """1-epoch linear warm-up then cosine decay to 0, evaluated per optimiser step."""
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


class InputNorm:
    """uint8 (B,H,W,3) RGB batch -> normalised float (B,3,H,W) channels_last tensor on the device.

    (x/255 - mean)/std is folded into one mul + one add. Doing this on the GPU keeps the
    DataLoader traffic at 1 byte/pixel and the workers free of float conversions."""

    def __init__(self, mean, std, device: torch.device):
        mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.scale = (1.0 / (255.0 * std)).to(device)
        self.shift = (-mean / std).to(device)
        self.device = device

    def __call__(self, imgs: torch.Tensor) -> torch.Tensor:
        x = imgs.to(self.device, non_blocking=True).permute(0, 3, 1, 2).float()  # NHWC -> NCHW view = channels_last
        x = x.contiguous(memory_format=torch.channels_last)
        return x.mul_(self.scale).add_(self.shift)


def make_loader(ds, batch: int, shuffle: bool, workers: int, drop_last: bool = False) -> DataLoader:
    kw = dict(batch_size=batch, shuffle=shuffle, num_workers=workers, pin_memory=True, drop_last=drop_last)
    if workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=4, worker_init_fn=wd.worker_init)
    return DataLoader(ds, **kw)


@torch.inference_mode()
def predict_probs(model: nn.Module, loader: DataLoader, norm: InputNorm, use_amp: bool, desc: str) -> np.ndarray:
    """Softmax probabilities (N, C) in loader order."""
    model.eval()
    out = []
    for imgs, _ in tqdm(loader, desc=desc, leave=False, unit="batch", mininterval=1.0):
        with autocast_ctx(use_amp):
            logits = model(norm(imgs))
        out.append(torch.softmax(logits.float(), dim=1))
    if not out:
        return np.zeros((0, len(CLASSES)), dtype=np.float32)
    return torch.cat(out).cpu().numpy()


# --------------------------------------------------------------------------- importable runtime API
def load_model(ckpt_path, device="cuda"):
    """Load a checkpoint written by `train`. Returns (model, meta); meta holds classes, img_size, mean, std,
    epoch, metrics, config, device, use_amp, the eval transform and the GPU input normaliser."""
    device = torch.device(device)
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model = build_model(num_classes=len(ck["classes"]), pretrained=False)
    model.load_state_dict(ck["model"])
    model.to(device, memory_format=torch.channels_last).eval()
    meta = {
        "classes": list(ck["classes"]), "img_size": int(ck["img_size"]),
        "mean": list(ck["mean"]), "std": list(ck["std"]),
        "epoch": ck.get("epoch"), "metrics": ck.get("metrics"), "config": ck.get("config"),
        "device": device, "use_amp": device.type == "cuda",
        "transform": wd.EvalTransform(int(ck["img_size"])),
        "norm": InputNorm(ck["mean"], ck["std"], device),
    }
    return model, meta


def predict_batch(model, meta, bgr_images) -> list[tuple[str, dict]]:
    """List of (label, {class: prob}) for a list of BGR uint8 HxWx3 arrays (any resolution)."""
    x = torch.from_numpy(np.stack([meta["transform"](img) for img in bgr_images]))
    with torch.inference_mode(), autocast_ctx(meta["use_amp"]):
        logits = model(meta["norm"](x))
    probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
    classes = meta["classes"]
    return [(classes[int(p.argmax())], {c: float(v) for c, v in zip(classes, p)}) for p in probs]


def predict_image(model, meta, bgr_ndarray) -> tuple[str, dict]:
    """(label, {class: prob}) for one BGR uint8 frame straight from cv2."""
    return predict_batch(model, meta, [bgr_ndarray])[0]


# --------------------------------------------------------------------------- manifest / cache glue
def prepare_manifest(args) -> pd.DataFrame:
    args.out.mkdir(parents=True, exist_ok=True)
    df = wd.build_manifest(args.root, args.out, args.bdd_labels)
    info = wd.check_leakage(df)
    tab = wd.counts_table(df)
    print("[manifest] per-(source, split, label) counts:\n" + tab.to_string())
    tab.to_csv(args.out / "counts.csv")
    df.to_csv(args.out / "manifest.csv", index=False)
    info.update({"rows": int(len(df)), "sources": sorted(df["source"].unique().tolist()),
                 "bdd_labels_used": bool((df["source"] == "bdd").any()),
                 "bdd_signature": wd.bdd_signature(args.root, args.bdd_labels)})
    wd.save_json(info, args.out / "manifest_info.json")
    print(f"[manifest] wrote {args.out / 'manifest.csv'} ({len(df)} rows) and counts.csv")
    return df


def get_manifest(args, build_if_missing: bool) -> pd.DataFrame:
    """Load <out>/manifest.csv; rebuild it when missing (train) or when the BDD label files on disk changed."""
    mp = args.out / "manifest.csv"
    if mp.exists():
        info_p = args.out / "manifest_info.json"
        stored = json.loads(info_p.read_text()).get("bdd_signature") if info_p.exists() else None
        if stored != wd.bdd_signature(args.root, args.bdd_labels):
            if not build_if_missing:
                sys.exit(f"[manifest] the BDD label files changed since {mp} was built - run prepare again")
            print("[manifest] the BDD label files changed since the manifest was built - rebuilding it")
            return prepare_manifest(args)
        df = wd.load_manifest(mp)
        print(f"[manifest] loaded {mp} ({len(df)} rows)")
        wd.check_leakage(df)
        return df
    if not build_if_missing:
        sys.exit(f"[manifest] {mp} not found - run: python {Path(__file__).name} prepare")
    print("[manifest] not found - running prepare first")
    return prepare_manifest(args)


def ensure_cache_rows(df: pd.DataFrame, args) -> None:
    wd.ensure_cache(df["path"].tolist(), df["cache_path"].tolist(), args.out, args.cache_size, args.cache_workers)


def labels_of(df: pd.DataFrame) -> np.ndarray:
    return df["label"].map(wd.CLASS_TO_IDX).to_numpy(dtype=np.int64)


# --------------------------------------------------------------------------- commands
def cmd_prepare(args) -> None:
    df = prepare_manifest(args)
    sub = wd.apply_limit(df, args.limit_per_class, args.seed)
    if args.limit_per_class:
        print(f"[prepare] --limit-per-class {args.limit_per_class}: caching {len(sub)} of {len(df)} rows")
    ensure_cache_rows(sub, args)
    print(f"[prepare] done - cache at {args.out / 'cache'}")


def train_one_epoch(model, loader, criterion, optimizer, norm, use_amp, base_lr, warmup, total, step, desc):
    model.train()
    loss_sum = torch.zeros((), device=norm.device)
    n = 0
    for imgs, labels in tqdm(loader, desc=desc, leave=False, unit="batch", mininterval=1.0):
        f = lr_factor(step, warmup, total)
        for g in optimizer.param_groups:
            g["lr"] = base_lr * g["lr_mult"] * f
        x = norm(imgs)
        y = labels.to(norm.device, non_blocking=True)
        with autocast_ctx(use_amp):
            logits = model(x)
        loss = criterion(logits.float(), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_sum += loss.detach() * y.shape[0]  # no per-step .item() sync
        n += y.shape[0]
        step += 1
    return loss_sum.item() / max(n, 1), step


def _val_f1(h: dict) -> float:
    return h["val_macro_f1"]


def make_ckpt(model, args, epoch, metrics) -> dict:
    return {
        "arch": "mobilenet_v3_large", "model": model.state_dict(), "classes": CLASSES,
        "img_size": args.img_size, "mean": list(wd.IMAGENET_MEAN), "std": list(wd.IMAGENET_STD),
        "epoch": epoch, "metrics": metrics, "config": json.loads(json.dumps(vars(args), default=str)),
    }


def cmd_train(args) -> None:
    device = setup_device(args.cpu)
    use_amp = device.type == "cuda"
    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    print("[config] " + json.dumps(vars(args), default=str))

    df = wd.apply_limit(get_manifest(args, build_if_missing=True), args.limit_per_class, args.seed)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[(df["split"] == "val") & (df["source"] == "acdc")].reset_index(drop=True)  # selection: ACDC val only
    if train_df.empty or val_df.empty:
        sys.exit(f"[train] empty train ({len(train_df)}) or ACDC val ({len(val_df)}) set")
    print(f"[data] train {len(train_df)} rows " + str(train_df["label"].value_counts().to_dict())
          + f" | ACDC val {len(val_df)} rows " + str(val_df["label"].value_counts().to_dict()))
    ensure_cache_rows(pd.concat([train_df, val_df]), args)

    y_train, y_val = labels_of(train_df), labels_of(val_df)
    train_ds = wd.CachedDataset(train_df["cache_path"], y_train,
                                wd.TrainTransform(args.img_size, scale=RRC_SCALE, ratio=RRC_RATIO))
    val_ds = wd.CachedDataset(val_df["cache_path"], y_val, wd.EvalTransform(args.img_size))
    drop_last = len(train_ds) >= 2 * args.batch
    train_loader = make_loader(train_ds, args.batch, True, args.workers, drop_last)
    val_loader = make_loader(val_ds, args.eval_batch, False, min(args.workers, 3))
    print(f"[aug] train: RandomResizedCrop({args.img_size}, scale={RRC_SCALE}, ratio={RRC_RATIO}) + "
          f"RandomHorizontalFlip only (no photometric ops) | eval: Resize({args.img_size}) + CenterCrop | "
          f"ImageNet mean/std applied on the GPU")

    counts = np.bincount(y_train, minlength=len(CLASSES)).astype(np.float64)
    inv = np.where(counts > 0, 1.0 / np.maximum(counts, 1), 0.0)
    weights = np.where(counts > 0, inv / inv[counts > 0].mean(), 1.0)  # inverse frequency, mean 1 over present classes
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device), label_smoothing=0.05)
    print("[loss] class-weighted CrossEntropyLoss (inverse frequency, normalised to mean 1) with label_smoothing=0.05; "
          "weights: " + ", ".join(f"{c}={w:.3f}" for c, w in zip(CLASSES, weights)))

    model = build_model(pretrained=True).to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(param_groups(model, args.lr, WEIGHT_DECAY, BACKBONE_LR_MULT),
                                  lr=args.lr, betas=(0.9, 0.999), fused=(device.type == "cuda"))
    steps_per_epoch = len(train_loader)
    warmup, total = steps_per_epoch, args.epochs * steps_per_epoch
    print(f"[opt] AdamW(wd={WEIGHT_DECAY}, fused={device.type == 'cuda'}) | head lr {args.lr:g}, backbone lr "
          f"{args.lr * BACKBONE_LR_MULT:g} | cosine over {total} steps with {warmup}-step (1 epoch) warm-up | "
          f"bf16 autocast={use_amp}, no GradScaler | channels_last | cudnn.benchmark")
    norm = InputNorm(wd.IMAGENET_MEAN, wd.IMAGENET_STD, device)

    start_epoch, step, best_f1, history = 0, 0, -1.0, []
    last_path, best_path = args.out / "last.pt", args.out / "best.pt"
    if args.resume:
        if last_path.exists():
            ck = torch.load(last_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            start_epoch, step, best_f1, history = ck["epoch"] + 1, ck["step"], ck["best_f1"], ck["history"]
            print(f"[resume] from {last_path}: epoch {ck['epoch'] + 1} done, best macro-F1 {best_f1:.4f}")
        else:
            sys.exit(f"[resume] {last_path} not found - check --out, or drop --resume to start from scratch")

    epoch_times, t_start = [], time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        t0 = time.perf_counter()
        train_loss, step = train_one_epoch(model, train_loader, criterion, optimizer, norm, use_amp, args.lr,
                                           warmup, total, step, f"epoch {epoch + 1}/{args.epochs}")
        probs = predict_probs(model, val_loader, norm, use_amp, "val")
        m = wd.classification_metrics(y_val, probs.argmax(1))
        secs = time.perf_counter() - t0
        epoch_times.append(secs)
        eta = float(np.mean(epoch_times)) * (args.epochs - epoch - 1)
        lr_now = args.lr * lr_factor(max(step - 1, 0), warmup, total)
        is_best = m["macro_f1"] > best_f1
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_acc": m["accuracy"],
                        "val_macro_f1": m["macro_f1"], "lr": lr_now, "epoch_seconds": secs, "is_best": is_best})
        pd.DataFrame(history).to_csv(args.out / "history.csv", index=False)
        print(f"[epoch {epoch + 1:02d}/{args.epochs}] loss {train_loss:.4f} | val acc {m['accuracy']:.4f} | "
              f"val macro-F1 {m['macro_f1']:.4f} | lr {lr_now:.2e} | {secs:.1f}s | ETA {eta / 60:.1f} min"
              + (" | new best" if is_best else ""))
        ckpt = make_ckpt(model, args, epoch, {"val_acdc": m, "train_loss": train_loss})
        if is_best:
            best_f1 = m["macro_f1"]
            torch.save(ckpt, best_path)
        ckpt.update({"optimizer": optimizer.state_dict(), "step": step, "best_f1": best_f1, "history": history})
        torch.save(ckpt, last_path)

    if history:
        wd.plot_training_curves(pd.DataFrame(history), args.out / "training_curves.png")
        best = max(history, key=_val_f1)  # first epoch reaching the top score = the one saved as best.pt
        print(f"[train] done in {(time.perf_counter() - t_start) / 60:.1f} min | best epoch {best['epoch']} "
              f"val macro-F1 {best['val_macro_f1']:.4f} acc {best['val_acc']:.4f} | {best_path}")
    del train_loader, val_loader  # stops the persistent worker processes and frees their RAM
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    if not args.no_final_eval and best_path.exists():
        print("[train] running eval on best.pt (pass --no-final-eval to skip)")
        args.ckpt = best_path
        cmd_eval(args)


def cmd_eval(args) -> None:
    device = setup_device(args.cpu)
    set_seed(args.seed)
    df = wd.apply_limit(get_manifest(args, build_if_missing=False), args.limit_per_class, args.seed)
    ev = df[df["split"].isin(["val", "test"])].reset_index(drop=True)
    ensure_cache_rows(ev, args)
    ckpt = args.ckpt or args.out / "best.pt"
    if not Path(ckpt).exists():
        sys.exit(f"[eval] checkpoint not found: {ckpt} - run train first")
    model, meta = load_model(ckpt, device)
    print(f"[eval] {ckpt} (epoch {meta['epoch']}, img_size {meta['img_size']}) on {len(ev)} val/test images")

    y_true = labels_of(ev)
    ds = wd.CachedDataset(ev["cache_path"], y_true, meta["transform"])
    probs = predict_probs(model, make_loader(ds, args.eval_batch, False, args.eval_workers), meta["norm"],
                          meta["use_amp"], "eval")
    y_pred = probs.argmax(1)
    ev["pred"] = [CLASSES[i] for i in y_pred]
    for i, c in enumerate(CLASSES):
        ev[f"p_{c}"] = probs[:, i]
    ev.to_csv(args.out / "eval_predictions.csv", index=False)

    real = ev["source"].isin(wd.REAL_SOURCES).to_numpy()
    metrics = {
        "checkpoint": str(ckpt), "epoch": meta["epoch"], "img_size": meta["img_size"], "classes": CLASSES,
        "note": "val/test blocks cover real sources (acdc + bdd when labelled); the synthetic foggy_cs probe "
                "appears only under per_source.test and in foggy_cityscapes_probe.json; macro-F1 is over classes "
                "present in the evaluated subset",
        "per_source": {},
    }
    for split in ("val", "test"):
        mask = real & (ev["split"] == split).to_numpy()
        m = wd.classification_metrics(y_true[mask], y_pred[mask])
        metrics[split] = m
        if m["n"]:
            wd.plot_confusion_matrix(m["confusion_matrix"], CLASSES,
                                     f"{split}: n={m['n']}  acc={m['accuracy']:.3f}  macro-F1={m['macro_f1']:.3f}",
                                     args.out / f"confusion_matrix_{split}.png")
            print(f"[eval] {split:4s} n={m['n']:5d} acc={m['accuracy']:.4f} macro-F1={m['macro_f1']:.4f} | "
                  + " ".join(f"{c}:F1={m['per_class'][c]['f1']:.3f}" for c in CLASSES))
        metrics["per_source"][split] = {}
        for src in sorted(ev.loc[ev["split"] == split, "source"].unique()):
            smask = ((ev["split"] == split) & (ev["source"] == src)).to_numpy()
            sm = wd.classification_metrics(y_true[smask], y_pred[smask])
            metrics["per_source"][split][src] = sm
            print(f"[eval]   {split}/{src}: n={sm['n']} acc={sm['accuracy']:.4f} macro-F1={sm['macro_f1']:.4f}")
    bmask = ((ev["split"] == "val") & (ev["source"] == "bdd")).to_numpy()
    if bmask.any():
        metrics["bdd_val"] = wd.classification_metrics(y_true[bmask], y_pred[bmask])
    wd.save_json(metrics, args.out / "metrics.json")

    fog = ev[ev["source"] == "foggy_cs"]
    probe = {"n": int(len(fog)), "checkpoint": str(ckpt), "per_beta": {}, "overall": None}
    if len(fog):
        def block(rows: pd.DataFrame) -> dict:
            return {"n": int(len(rows)), "fraction_pred_fog": float((rows["pred"] == "fog").mean()),
                    "pred_distribution": {c: int((rows["pred"] == c).sum()) for c in CLASSES}}
        for beta in sorted(fog["extra"].unique(), key=float):
            probe["per_beta"][beta] = block(fog[fog["extra"] == beta])
        probe["overall"] = block(fog)
        refined_file = args.root / wd.FOGGY_REFINED
        if refined_file.exists():
            refined = {ln.strip() for ln in refined_file.read_text().splitlines() if ln.strip()}
            cs_split = fog["path"].str.split("/").str[-3]  # train / val folder of Cityscapes
            key = cs_split + "/" + fog["group"] + "/" + fog["group"] + "_" + fog["frame"]
            rf = fog[key.isin(refined)]
            probe["refined_subset"] = {"n": int(len(rf)), "per_beta": {b: block(rf[rf["extra"] == b])
                                                                      for b in sorted(rf["extra"].unique(), key=float)}}
        print("[eval] foggy Cityscapes probe: " + ", ".join(
            f"beta {b}: fog {v['fraction_pred_fog']:.3f} (n={v['n']})" for b, v in probe["per_beta"].items()))
    wd.save_json(probe, args.out / "foggy_cityscapes_probe.json")

    hist = args.out / "history.csv"
    if hist.exists():
        wd.plot_training_curves(pd.read_csv(hist), args.out / "training_curves.png")
    print(f"[eval] wrote metrics.json, confusion_matrix_{{val,test}}.png, foggy_cityscapes_probe.json, "
          f"eval_predictions.csv under {args.out}")


def cmd_predict(args) -> None:
    device = setup_device(args.cpu)
    ckpt = args.ckpt or args.out / "best.pt"
    if not Path(ckpt).exists():
        sys.exit(f"[predict] checkpoint not found: {ckpt} - run train first")
    model, meta = load_model(ckpt, device)
    paths = wd.list_images(args.predict_dir)
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        sys.exit(f"[predict] no jpg/png images under {args.predict_dir}")
    ds = wd.ImageFileDataset(paths, meta["transform"], min_short_side=meta["img_size"])
    t0 = time.perf_counter()
    probs = predict_probs(model, make_loader(ds, args.eval_batch, False, args.eval_workers), meta["norm"],
                          meta["use_amp"], "predict")
    secs = time.perf_counter() - t0
    pred = [CLASSES[i] for i in probs.argmax(1)]
    stems = [p.stem for p in paths]
    if len(set(stems)) != len(stems):
        print(f"[predict] warning: {len(stems) - len(set(stems))} duplicate frame_id stems under {args.predict_dir}")
    table = pd.DataFrame({"frame_id": stems, "weather_pred": pred})
    for i, c in enumerate(CLASSES):
        table[f"p_{c}"] = probs[:, i]
    args.out.mkdir(parents=True, exist_ok=True)
    csv = args.out / f"predictions_{Path(args.predict_dir).name}.csv"
    table.to_csv(csv, index=False, float_format="%.5f")
    print(f"[predict] {len(paths)} images in {secs:.1f}s ({len(paths) / max(secs, 1e-6):.0f} img/s) -> {csv}")
    print("[predict] class distribution: " + str(table["weather_pred"].value_counts().to_dict()))


def main(argv=None) -> None:
    args = parse_args(argv)
    wd.start_log(args.out / f"log_{args.cmd}.txt")
    {"prepare": cmd_prepare, "train": cmd_train, "eval": cmd_eval, "predict": cmd_predict}[args.cmd](args)


if __name__ == "__main__":
    main()
