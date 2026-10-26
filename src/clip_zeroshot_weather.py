"""CLIP zero-shot weather baseline (open_clip ViT-B-32, 'openai' weights) on the classifier's manifest.

Reads <out>/manifest.csv written by `train_weather_classifier.py prepare`, evaluates the val and
test splits from the same resized cache with a small prompt ensemble per class, and writes
    <out>/clip_zeroshot/metrics_clip.json
    <out>/clip_zeroshot/confusion_matrix_clip_val.png / confusion_matrix_clip_test.png
    <out>/clip_zeroshot/predictions_clip.csv
Same conventions as the trained model's eval: val/test blocks cover the real sources (acdc, bdd),
the synthetic foggy_cs rows are reported as a per-beta probe.

Usage (CUDA env active, project root as working directory):
    pip install open_clip_torch                      # once; first run downloads the 605 MB ViT-B-32 weights from Hugging Face
    python src/clip_zeroshot_weather.py [--limit-per-class 16] [--cpu]   (console output also -> <out>/clip_zeroshot/log_clip.txt)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# huggingface_hub's Xet transfer backend stalled on this network (18 Sept 2026) while plain HTTP worked,
# so default to plain HTTP for the one-off weight download. Must be set before open_clip is imported.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import weather_data as wd  # noqa: E402

CLASSES = wd.CLASSES
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
PROMPTS = {
    "clear": [
        "a photo of a road in clear weather",
        "a dashcam photo of a street on a clear sunny day",
        "a photo of a city street in good weather during the day",
        "a photo of a road under a clear sky",
    ],
    "fog": [
        "a dashcam photo of a foggy road",
        "a photo of a street covered in thick fog",
        "a photo of a road with low visibility because of fog",
        "a photo of a hazy, misty road",
    ],
    "night": [
        "a photo of a road at night",
        "a dashcam photo of a dark street at night with headlights and street lights",
        "a photo of a city street after dark",
        "a photo of a road in the dark",
    ],
    "rain": [
        "a photo of a road in the rain",
        "a dashcam photo of a wet street while it is raining",
        "a photo of a rainy street with raindrops on the windshield",
        "a photo of a road with puddles during rainfall",
    ],
    "snow": [
        "a photo of a snowy road",
        "a dashcam photo of a street covered in snow",
        "a photo of a road while it is snowing",
        "a photo of a winter road with snow on the ground",
    ],
}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root", type=Path, default=SCRIPT_DIR.parent)
    p.add_argument("--out", type=Path, default=Path("results/weather_classifier"),
                   help="folder holding manifest.csv and the cache; outputs go to <out>/clip_zeroshot")
    p.add_argument("--model", default="ViT-B-32-quickgelu",
                   help="the OpenAI weights were trained with QuickGELU; plain ViT-B-32 loads them into the wrong activation")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--img-size", type=int, default=224, help="CLIP input size")
    p.add_argument("--cache-size", type=int, default=320)
    p.add_argument("--cache-workers", type=int, default=8)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit-per-class", type=int, default=None, metavar="N")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args(argv)
    args.root = args.root.resolve()
    args.out = (args.out if args.out.is_absolute() else args.root / args.out).resolve()
    return args


class ClipNorm:
    """uint8 (B,H,W,3) RGB -> CLIP-normalised float (B,3,H,W) on the device (same trick as the classifier)."""

    def __init__(self, device: torch.device):
        mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
        self.scale, self.shift, self.device = (1.0 / (255.0 * std)).to(device), (-mean / std).to(device), device

    def __call__(self, imgs: torch.Tensor) -> torch.Tensor:
        x = imgs.to(self.device, non_blocking=True).permute(0, 3, 1, 2).float().contiguous()
        return x.mul_(self.scale).add_(self.shift)


def main(argv=None) -> None:
    args = parse_args(argv)
    wd.start_log(args.out / "clip_zeroshot" / "log_clip.txt")
    try:
        import open_clip
    except ImportError:
        sys.exit("open_clip is missing: pip install open_clip_torch")
    print(f"[env] torch {torch.__version__} | open_clip {open_clip.__version__} | CUDA build {torch.version.cuda}")
    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[env] GPU {torch.cuda.get_device_name(0)}")
    else:
        sys.exit("[env] CUDA not available; install the cu128 torch wheels or pass --cpu")
    torch.manual_seed(args.seed)
    use_amp = device.type == "cuda"

    manifest = args.out / "manifest.csv"
    if not manifest.exists():
        sys.exit(f"{manifest} not found - run `python train_weather_classifier.py prepare` first")
    df = wd.apply_limit(wd.load_manifest(manifest), args.limit_per_class, args.seed)
    ev = df[df["split"].isin(["val", "test"])].reset_index(drop=True)
    wd.ensure_cache(ev["path"].tolist(), ev["cache_path"].tolist(), args.out, args.cache_size, args.cache_workers)
    out_dir = args.out / "clip_zeroshot"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(args.model)
    model.eval()
    print(f"[clip] {args.model}/{args.pretrained}; prompt ensemble: " + json.dumps(PROMPTS))
    with torch.inference_mode():
        text_feats = []
        for c in CLASSES:
            f = model.encode_text(tokenizer(PROMPTS[c]).to(device)).float()
            text_feats.append(F.normalize(F.normalize(f, dim=-1).mean(0), dim=0))
        text_feats = torch.stack(text_feats)  # (5, D)
        logit_scale = float(model.logit_scale.exp())

    # Resize(224)+CenterCrop(224) via cv2 (area filter instead of PIL bicubic), CLIP mean/std on the GPU
    y_true = ev["label"].map(wd.CLASS_TO_IDX).to_numpy(dtype=np.int64)
    ds = wd.CachedDataset(ev["cache_path"], y_true, wd.EvalTransform(args.img_size))
    kw = dict(batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    if args.workers > 0:
        kw.update(prefetch_factor=4, worker_init_fn=wd.worker_init)
    loader = DataLoader(ds, **kw)
    norm = ClipNorm(device)
    probs = []
    t0 = time.perf_counter()
    with torch.inference_mode():
        for imgs, _ in tqdm(loader, desc="clip", unit="batch", leave=False, mininterval=1.0):
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                f = model.encode_image(norm(imgs))
            f = F.normalize(f.float(), dim=-1)
            probs.append(torch.softmax(logit_scale * f @ text_feats.T, dim=1))
    probs = torch.cat(probs).cpu().numpy()
    print(f"[clip] {len(ev)} images in {time.perf_counter() - t0:.1f}s")
    y_pred = probs.argmax(1)
    ev["pred"] = [CLASSES[i] for i in y_pred]
    for i, c in enumerate(CLASSES):
        ev[f"p_{c}"] = probs[:, i]
    ev.to_csv(out_dir / "predictions_clip.csv", index=False)

    real = ev["source"].isin(wd.REAL_SOURCES).to_numpy()
    metrics = {"model": f"{args.model}/{args.pretrained}", "prompts": PROMPTS, "classes": CLASSES,
               "note": "val/test blocks cover real sources (acdc + bdd if labelled); foggy_cs only under "
                       "per_source.test and foggy_cityscapes_probe", "per_source": {}}
    for split in ("val", "test"):
        mask = real & (ev["split"] == split).to_numpy()
        m = wd.classification_metrics(y_true[mask], y_pred[mask])
        metrics[split] = m
        if m["n"]:
            wd.plot_confusion_matrix(m["confusion_matrix"], CLASSES,
                                     f"CLIP zero-shot {split}: n={m['n']}  acc={m['accuracy']:.3f}  "
                                     f"macro-F1={m['macro_f1']:.3f}", out_dir / f"confusion_matrix_clip_{split}.png")
            print(f"[clip] {split:4s} n={m['n']:5d} acc={m['accuracy']:.4f} macro-F1={m['macro_f1']:.4f} | "
                  + " ".join(f"{c}:F1={m['per_class'][c]['f1']:.3f}" for c in CLASSES))
        metrics["per_source"][split] = {}
        for src in sorted(ev.loc[ev["split"] == split, "source"].unique()):
            smask = ((ev["split"] == split) & (ev["source"] == src)).to_numpy()
            metrics["per_source"][split][src] = wd.classification_metrics(y_true[smask], y_pred[smask])
    bmask = ((ev["split"] == "val") & (ev["source"] == "bdd")).to_numpy()
    if bmask.any():
        metrics["bdd_val"] = wd.classification_metrics(y_true[bmask], y_pred[bmask])
    fog = ev[ev["source"] == "foggy_cs"]
    metrics["foggy_cityscapes_probe"] = {
        beta: {"n": int(len(rows)), "fraction_pred_fog": float((rows["pred"] == "fog").mean()),
               "pred_distribution": {c: int((rows["pred"] == c).sum()) for c in CLASSES}}
        for beta, rows in ((b, fog[fog["extra"] == b]) for b in sorted(fog["extra"].unique(), key=float))
    }
    wd.save_json(metrics, out_dir / "metrics_clip.json")
    print(f"[clip] wrote metrics_clip.json, confusion_matrix_clip_{{val,test}}.png, predictions_clip.csv -> {out_dir}")


if __name__ == "__main__":
    main()
