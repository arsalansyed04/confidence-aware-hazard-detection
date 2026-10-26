"""Weather classifier v4: condition + fog level + rain level, leave-one-camera-out benchmark, 2-model ensemble.

Separate experiment: writes only to results/weather_v4 (plus resized JPEGs and depth maps in the shared cache).
results/weather_v3 (the v3 model, trained before the full BDD100K images existed) is kept as the fallback and is a comparison column.
See src_v4/README.md for the design and the reasons behind it.

Commands (Anaconda PowerShell Prompt, conda env "weather", project root as working directory):
    python src_v4/train_weather_v4.py prepare    # manifest, split tables, image cache
    python src_v4/train_weather_v4.py depth      # depth maps for the clear images + fog-renderer calibration
    python src_v4/train_weather_v4.py lodo       # leave-one-camera-out benchmark of the finalist recipes
    python src_v4/train_weather_v4.py train      # trains the 2 best recipes of different backbones, then evaluates the ensemble
    python src_v4/train_weather_v4.py eval       # ensemble evaluation + comparison with v2 and the earlier v3
    python src_v4/train_weather_v4.py predict --predict-dir bdd100k-10k/val
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import weather_v4_lib as L  # noqa: E402

wd, v1, v2 = L.wd, L.v1, L.v2
CLASSES = L.CLASSES


# --------------------------------------------------------------------------- CLI
def parse_args(argv=None) -> argparse.Namespace:
    c = argparse.ArgumentParser(add_help=False)
    c.add_argument("--root", type=Path, default=L.PROJECT_ROOT)
    c.add_argument("--out", type=Path, default=Path("results/weather_v4"))
    c.add_argument("--cache-root", type=Path, default=Path("results/weather_classifier"),
                   help="shared cache of resized JPEGs and depth maps (only cache files are written there)")
    c.add_argument("--v2-ckpt", type=Path, default=Path("results/weather_classifier_cityscapes/best.pt"))
    c.add_argument("--old-v3-ckpt", type=Path, default=Path("results/weather_v3/best.pt"),
                   help="the v3 model trained before the full BDD100K images existed (comparison column)")
    c.add_argument("--bdd100k-images", type=Path, default=None,
                   help="folder holding train/ and val/ of the full BDD100K images (auto-detected under --root)")
    c.add_argument("--bdd-labels", type=str, default=None)
    c.add_argument("--bdd-cap", type=int, default=100000,
                   help="max BDD100K training images per class; the default is effectively no cap (the largest class has "
                        "about 26,600). Lower it only to shorten caching")
    c.add_argument("--render-train-cap", type=int, default=8000, help="max clear images per source used for rendering (train)")
    c.add_argument("--render-eval-cap", type=int, default=600, help="same, per source, for val and test")
    c.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Small-hf", help="or 'prior' (no network)")
    c.add_argument("--cache-size", type=int, default=320)
    c.add_argument("--cache-workers", type=int, default=12)
    c.add_argument("--workers", type=int, default=3, help="training loader processes (auto-capped by free memory)")
    c.add_argument("--eval-workers", type=int, default=2)
    c.add_argument("--eval-batch", type=int, default=128)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--limit-per-class", type=int, default=None, metavar="N", help="smoke test")
    c.add_argument("--no-pretrained", action="store_true", help="random weights (code-path tests only)")
    c.add_argument("--no-camera-aug", action="store_true")
    c.add_argument("--no-tta", action="store_true", help="skip the mirrored-image averaging at eval / predict time")
    c.add_argument("--cpu", action="store_true")

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("prepare", parents=[c])
    pp.add_argument("--manifest-only", action="store_true")
    dp = sub.add_parser("depth", parents=[c])
    dp.add_argument("--calibration-pairs", type=int, default=100)
    lo = sub.add_parser("lodo", parents=[c], help="leave-one-camera-out benchmark")
    lo.add_argument("--configs", nargs="+", default=list(L.FINALISTS), choices=list(L.CONFIGS))
    lo.add_argument("--epochs", type=int, default=6)
    lo.add_argument("--samples-per-epoch", type=int, default=10000)
    t = sub.add_parser("train", parents=[c])
    t.add_argument("--configs", nargs="+", default=None, choices=list(L.CONFIGS),
                   help="recipes to train (default: best LODO recipe + best recipe of a different backbone)")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--samples-per-epoch", type=int, default=24000)
    t.add_argument("--no-final-eval", action="store_true")
    e = sub.add_parser("eval", parents=[c])
    e.add_argument("--ckpts", nargs="+", type=Path, default=None, help="default: every <out>/ckpt_*.pt (the ensemble)")
    pr = sub.add_parser("predict", parents=[c])
    pr.add_argument("--ckpts", nargs="+", type=Path, default=None)
    pr.add_argument("--predict-dir", type=Path, required=True)
    pr.add_argument("--limit", type=int, default=None)

    a = p.parse_args(argv)
    a.root = a.root.resolve()
    for name in ("out", "cache_root", "v2_ckpt", "old_v3_ckpt"):
        v = getattr(a, name)
        setattr(a, name, (v if v.is_absolute() else a.root / v).resolve())
    if a.bdd100k_images is not None:
        a.bdd100k_images = (a.bdd100k_images if a.bdd100k_images.is_absolute() else a.root / a.bdd100k_images).resolve()
    return a


# --------------------------------------------------------------------------- data plumbing
def prepare_manifest(args) -> pd.DataFrame:
    args.out.mkdir(parents=True, exist_ok=True)
    df = L.build_manifest(args)
    L.check_manifest(df)
    real = df[df["render"] == ""]
    print("[manifest] REAL images per (source, split, label):\n" + L.counts(real).to_string())
    virt = df[df["render"] != ""]
    print("[manifest] rendered-on-the-fly rows (fog / rain drawn onto clear images, by base camera):\n" + L.counts(virt).to_string())
    lv = real[(real["fog_level"] > 0) | (real["rain_level"] > 0)]
    print(f"[manifest] real rows with a known severity: fog {int((real['fog_level'] > 0).sum())}, rain {int((real['rain_level'] > 0).sum())} "
          f"(sources: {sorted(lv['source'].unique())}); clear rows labelled 'none': {int((real['fog_level'] == 0).sum())}")
    L.counts(real).to_csv(args.out / "counts_real.csv")
    L.counts(virt).to_csv(args.out / "counts_rendered.csv")
    df.to_csv(args.out / "manifest.csv", index=False)
    L.save_json({"rows": len(df), "sources": sorted(df["source"].unique()), "bdd_signature": wd.bdd_signature(args.root, args.bdd_labels),
                 "test_cities": v2.CS_TEST_CITIES, "val_cities": v2.CS_VAL_CITIES}, args.out / "manifest_info.json")
    print(f"[manifest] wrote {args.out / 'manifest.csv'} ({len(df)} rows)")
    return df


def get_manifest(args) -> pd.DataFrame:
    mp = args.out / "manifest.csv"
    if not mp.exists():
        print("[manifest] not found - building it first")
        df = prepare_manifest(args)
    else:
        df = L.load_manifest(mp)
        print(f"[manifest] loaded {mp} ({len(df)} rows)")
        L.check_manifest(df)
    return wd.apply_limit(df, args.limit_per_class, args.seed)


def ensure_cache(df: pd.DataFrame, args) -> None:
    u = df.drop_duplicates("cache_path")
    wd.ensure_cache(u["path"].tolist(), u["cache_path"].tolist(), args.cache_root, args.cache_size, args.cache_workers)


def ensure_depth(df: pd.DataFrame) -> None:
    need = df.loc[df["render"] != "", "depth_path"].unique()
    missing = [p for p in need if not os.path.exists(p)]
    if missing:
        sys.exit(f"[depth] {len(missing)} of {len(need)} depth maps are missing - run:  python src_v4/train_weather_v4.py depth")


def fog_params(args) -> dict:
    f = args.out / "fog_render_params.json"
    if f.exists():
        return json.loads(f.read_text())
    sys.exit(f"[render] {f} not found - run:  python src_v4/train_weather_v4.py depth")


def loader_for(ds, batch, workers, sampler=None, drop_last=False) -> DataLoader:
    kw = dict(batch_size=batch, sampler=sampler, shuffle=False, num_workers=workers, pin_memory=True, drop_last=drop_last)
    if workers > 0:
        kw.update(persistent_workers=sampler is not None, prefetch_factor=4, worker_init_fn=wd.worker_init)
    return DataLoader(ds, **kw)


def find_ckpts(args) -> list[Path]:
    given = getattr(args, "ckpts", None)
    if given:
        return [(p if p.is_absolute() else args.root / p).resolve() for p in given]
    found = sorted(args.out.glob("ckpt_*.pt"))
    return found or ([args.out / "best.pt"] if (args.out / "best.pt").exists() else [])


# --------------------------------------------------------------------------- inference and metrics
@torch.inference_mode()
def infer(model, df: pd.DataFrame, cfg: dict, args, device, params: dict, workers: int, desc: str, tta: bool = False) -> dict:
    """Softmax outputs of the three heads in row order. tta=True also averages in the left-right mirrored image."""
    ds = L.V3Dataset(df, cfg["img"], False, params)
    norm = v1.InputNorm(wd.IMAGENET_MEAN, wd.IMAGENET_STD, device)
    model.eval()
    acc = {k: [] for k in ("pc", "pf", "pr", "y", "yf", "yr", "idx")}
    for imgs, y, yf, yr, idx in tqdm(loader_for(ds, args.eval_batch, workers), desc=desc, leave=False, unit="batch", mininterval=1.0):
        x = norm(imgs)
        with v1.autocast_ctx(device.type == "cuda"):
            outs = [model(x)] + ([model(x.flip(3))] if tta else [])
        for j, k in enumerate(("pc", "pf", "pr")):
            acc[k].append(torch.stack([torch.softmax(o[j].float(), 1) for o in outs]).mean(0).cpu().numpy())
        for k, v in (("y", y), ("yf", yf), ("yr", yr), ("idx", idx)):
            acc[k].append(v.numpy())
    out = {k: np.concatenate(v) if v else np.zeros((0,)) for k, v in acc.items()}
    order = np.argsort(out["idx"])
    return {k: v[order] for k, v in out.items()}


@torch.inference_mode()
def infer_files(model, meta: dict, paths: list, args, workers: int, desc: str, tta: bool) -> list[np.ndarray]:
    """[condition, fog-level, rain-level] softmax arrays for arbitrary image files (no labels needed)."""
    ds = wd.ImageFileDataset(paths, meta["transform"], min_short_side=meta["config"]["img"])
    parts = [[], [], []]
    for imgs, _ in tqdm(loader_for(ds, args.eval_batch, workers), desc=desc, unit="batch", leave=False, mininterval=1.0):
        x = meta["norm"](imgs)
        with v1.autocast_ctx(meta["use_amp"]):
            outs = [model(x)] + ([model(x.flip(3))] if tta else [])
        for j in range(3):
            parts[j].append(torch.stack([torch.softmax(o[j].float(), 1) for o in outs]).mean(0).cpu().numpy())
    return [np.concatenate(p) for p in parts]


def level_metrics(y: np.ndarray, p: np.ndarray, names: list[str]) -> dict:
    m = y >= 0
    if not m.any():
        return {"n": 0}
    y, p = y[m], p[m]
    from sklearn.metrics import confusion_matrix
    present = [k for k in range(len(names)) if (y == k).any()]
    rec = [float((p[y == k] == k).mean()) for k in present]
    return {"n": int(m.sum()), "accuracy": float((y == p).mean()), "balanced_accuracy": float(np.mean(rec)),
            "within_one_level": float((np.abs(y - p) <= 1).mean()), "per_level_recall": dict(zip([names[k] for k in present], rec)),
            "confusion_matrix": confusion_matrix(y, p, labels=list(range(len(names)))).tolist()}


def summarize(df: pd.DataFrame, out: dict) -> dict:
    pred = out["pc"].argmax(1)
    real = (df["render"] == "").to_numpy()
    s = {"condition_real": wd.classification_metrics(out["y"][real], pred[real]) if real.any() else {"n": 0},
         "condition_all": wd.classification_metrics(out["y"], pred),
         "fog_level": level_metrics(out["yf"], out["pf"].argmax(1), L.FOG_LEVELS),
         "rain_level": level_metrics(out["yr"], out["pr"].argmax(1), L.RAIN_LEVELS)}
    s["score"] = (0.6 * s["condition_all"].get("macro_f1", 0.0) + 0.2 * s["fog_level"].get("balanced_accuracy", 0.0)
                  + 0.2 * s["rain_level"].get("balanced_accuracy", 0.0))
    return s


# --------------------------------------------------------------------------- training
def fit(args, cfg_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame, device, epochs: int, samples: int, tag: str, params: dict):
    cfg = L.CONFIGS[cfg_name]
    use_amp = device.type == "cuda"
    v1.set_seed(args.seed)
    model = L.WeatherNetV3(cfg["backbone"], cfg["tune"], cfg["img"], pretrained=not args.no_pretrained).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{tag}] recipe {cfg_name}: {cfg} | trainable parameters {n_train / 1e6:.2f} M of {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    weights, share = L.sampling_weights(train_df)
    train_w, val_w = v2.cap_workers(args.workers, args.eval_workers, tag)
    ds = L.V3Dataset(train_df, cfg["img"], True, params, camera_augment=not args.no_camera_aug)
    n_draws = min(samples, max(cfg["batch"], len(ds) * 4))
    sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), n_draws, True, generator=torch.Generator().manual_seed(args.seed))
    loader = loader_for(ds, cfg["batch"], train_w, sampler, drop_last=n_draws >= 2 * cfg["batch"])
    opt = torch.optim.AdamW(L.param_groups(model, cfg["lr"], cfg["backbone_lr_mult"]), lr=cfg["lr"], fused=use_amp)
    steps = len(loader)
    warmup, total = steps, epochs * steps
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    ce_level = nn.CrossEntropyLoss(label_smoothing=0.05, ignore_index=-1)
    norm = v1.InputNorm(wd.IMAGENET_MEAN, wd.IMAGENET_STD, device)
    print(f"[{tag}] train pool {len(train_df)} rows, {n_draws} draws/epoch, {steps} steps/epoch, {epochs} epochs | val {len(val_df)} rows")

    best, best_state, history, step, times = -1.0, None, [], 0, []
    for epoch in range(epochs):
        t0 = time.perf_counter()
        model.train()
        loss_sum = torch.zeros((), device=device)
        seen = 0
        for imgs, y, yf, yr, _ in tqdm(loader, desc=f"{tag} {epoch + 1}/{epochs}", leave=False, unit="batch", mininterval=1.0):
            f = v1.lr_factor(step, warmup, total)
            for g in opt.param_groups:
                g["lr"] = cfg["lr"] * g["lr_mult"] * f
            y, yf, yr = y.to(device), yf.to(device), yr.to(device)
            with v1.autocast_ctx(use_amp):
                c, fo, ra = model(norm(imgs))
            loss = ce(c.float(), y)
            if (yf >= 0).any():
                loss = loss + 0.5 * ce_level(fo.float(), yf)
            if (yr >= 0).any():
                loss = loss + 0.5 * ce_level(ra.float(), yr)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.detach() * y.shape[0]
            seen += y.shape[0]
            step += 1
        s = summarize(val_df, infer(model, val_df, cfg, args, device, params, val_w, "val")) if len(val_df) else {"score": 0.0}
        secs = time.perf_counter() - t0
        times.append(secs)
        is_best = s["score"] > best
        if is_best:
            best = s["score"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append({"epoch": epoch + 1, "train_loss": loss_sum.item() / max(seen, 1), "val_score": s["score"],
                        "val_cond_macro_f1": s.get("condition_all", {}).get("macro_f1"), "val_cond_real_accuracy": s.get("condition_real", {}).get("accuracy"),
                        "val_fog_level_bal_acc": s.get("fog_level", {}).get("balanced_accuracy"),
                        "val_rain_level_bal_acc": s.get("rain_level", {}).get("balanced_accuracy"), "epoch_seconds": secs, "is_best": is_best})
        h = history[-1]
        print(f"[{tag} {epoch + 1:02d}/{epochs}] loss {h['train_loss']:.4f} | val cond macro-F1 {h['val_cond_macro_f1'] or 0:.4f} | fog-level bal-acc "
              f"{h['val_fog_level_bal_acc'] or 0:.4f} | rain-level bal-acc {h['val_rain_level_bal_acc'] or 0:.4f} | score {s['score']:.4f} | {secs:.0f}s | "
              f"ETA {np.mean(times) * (epochs - epoch - 1) / 60:.1f} min" + (" | new best" if is_best else ""))
    del loader
    gc.collect()
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, cfg, history, share


def free(device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- commands
def cmd_prepare(args) -> None:
    df = prepare_manifest(args)
    if args.manifest_only:
        print("[prepare] --manifest-only: no cache built")
        return
    ensure_cache(wd.apply_limit(df, args.limit_per_class, args.seed), args)
    print("[prepare] done. Next:  python src_v4/train_weather_v4.py depth")


def cmd_depth(args) -> None:
    device = v1.setup_device(args.cpu)
    df = get_manifest(args)
    ensure_cache(df[df["render"] != ""], args)
    L.compute_depth_maps(df, device, args.depth_model)
    pairs = L.calibration_pairs(L.load_manifest(args.out / "manifest.csv"), args.calibration_pairs, args.seed)
    if len(pairs):  # the real DBF fog frames of the calibration scenes must be in the cache too
        need = pd.concat([pd.DataFrame({"path": pairs[f"dbf_path_{b}"], "cache_path": pairs[f"dbf_cache_{b}"]}) for b in L.EVAL_BETAS])
        ensure_cache(need, args)
    params = L.calibrate_fog(pairs)
    params["depth_model"] = args.depth_model
    L.save_json(params, args.out / "fog_render_params.json")
    print(f"[depth] wrote {args.out / 'fog_render_params.json'}. Next:  python src_v4/train_weather_v4.py lodo")


def cmd_lodo(args) -> None:
    device = v1.setup_device(args.cpu)
    df = get_manifest(args)
    df = df[df["domain"].isin(L.TRAIN_DOMAINS)].reset_index(drop=True)
    ensure_cache(df, args)
    ensure_depth(df)
    params = fog_params(args)
    rows = []
    for cfg_name in args.configs:
        for held in L.TRAIN_DOMAINS:
            train_df = df[(df["split"] == "train") & (df["domain"] != held)].reset_index(drop=True)
            val_df = df[(df["split"] == "val") & (df["domain"] != held)].reset_index(drop=True)
            held_df = df[(df["domain"] == held) & df["split"].isin(["val", "test"])]
            if int((held_df["render"] == "").sum()) < 1000:  # small held-out camera: every image of it is unseen anyway
                held_df = df[df["domain"] == held]
            held_df = held_df.reset_index(drop=True)
            tag = f"lodo {cfg_name} -{held}"
            model, cfg, _, _ = fit(args, cfg_name, train_df, val_df, device, args.epochs, args.samples_per_epoch, tag, params)
            _, w = v2.cap_workers(0, args.eval_workers, tag + " eval")
            s = summarize(held_df, infer(model, held_df, cfg, args, device, params, w, "held-out"))
            rows.append({"config": cfg_name, "held_out_camera": held, "n_real": s["condition_real"].get("n", 0),
                         "cond_accuracy": s["condition_real"].get("accuracy"), "cond_macro_f1": s["condition_real"].get("macro_f1"),
                         "fog_level_bal_acc": s["fog_level"].get("balanced_accuracy"), "rain_level_bal_acc": s["rain_level"].get("balanced_accuracy")})
            print(f"[lodo] {cfg_name} on UNSEEN {held}: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items() if k not in ("config", "held_out_camera")}))
            pd.DataFrame(rows).to_csv(args.out / "lodo_results.csv", index=False)
            del model
            free(device)
    res = pd.DataFrame(rows)
    mean = res.groupby("config")[["cond_accuracy", "cond_macro_f1", "fog_level_bal_acc", "rain_level_bal_acc"]].mean()
    mean["selection"] = 0.6 * mean["cond_macro_f1"] + 0.2 * mean["fog_level_bal_acc"].fillna(0) + 0.2 * mean["rain_level_bal_acc"].fillna(0)
    mean = mean.sort_values("selection", ascending=False)
    print("[lodo] per run (model never saw the held-out camera):\n" + res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("[lodo] mean over the three held-out cameras:\n" + mean.to_string(float_format=lambda x: f"{x:.4f}"))
    mean.to_csv(args.out / "lodo_summary.csv")
    L.save_json({"best_config": mean.index[0], "summary": mean.reset_index().to_dict("records")}, args.out / "lodo_best.json")
    print(f"[lodo] ranking saved -> {args.out / 'lodo_best.json'}. Next:  python src_v4/train_weather_v4.py train")


def pick_configs(args) -> list[str]:
    """Best LODO recipe + the best recipe with a DIFFERENT backbone (an ensemble needs models that err differently)."""
    if args.configs:
        return list(args.configs)
    lb = args.out / "lodo_best.json"
    if not lb.exists():
        print("[train] no LODO ranking in this folder: using dinov2_last2 + convnext_full (the two best of the first LODO)")
        return ["dinov2_last2", "convnext_full"]
    ranked = [r["config"] for r in json.loads(lb.read_text())["summary"]]
    first = ranked[0]
    second = next((c for c in ranked[1:] if L.CONFIGS[c]["backbone"] != L.CONFIGS[first]["backbone"]), None)
    chosen = [first] + ([second] if second else [])
    print(f"[train] LODO ranking {ranked} -> training {chosen}")
    return chosen


def cmd_train(args) -> None:
    device = v1.setup_device(args.cpu)
    args.out.mkdir(parents=True, exist_ok=True)
    print("[config] " + json.dumps(vars(args), default=str))
    df = get_manifest(args)
    use = df[df["domain"].isin(L.TRAIN_DOMAINS)]
    train_df = use[use["split"] == "train"].reset_index(drop=True)
    val_df = use[use["split"] == "val"].reset_index(drop=True)
    ensure_cache(pd.concat([train_df, val_df]), args)
    ensure_depth(pd.concat([train_df, val_df]))
    params = fog_params(args)
    for cfg_name in pick_configs(args):
        model, cfg, history, share = fit(args, cfg_name, train_df, val_df, device, args.epochs, args.samples_per_epoch, f"train {cfg_name}", params)
        share.to_csv(args.out / "sampling_shares.csv")
        hist = pd.DataFrame(history)
        hist.to_csv(args.out / f"history_{cfg_name}.csv", index=False)
        best = hist.loc[hist["val_score"].idxmax()]
        L.save_checkpoint(args.out / f"ckpt_{cfg_name}.pt", model, cfg, cfg_name, params, {"epoch": int(best["epoch"]), "metrics": best.to_dict()})
        print(f"[train] {cfg_name}: best epoch {int(best['epoch'])} val score {best['val_score']:.4f} -> {args.out / f'ckpt_{cfg_name}.pt'}")
        del model
        free(device)
    print("[sampler] share of each epoch's draws:\n" + share.to_string(float_format=lambda x: f"{x:.3f}"))
    if not args.no_final_eval:
        args.ckpts = None
        cmd_eval(args, device)


def headline(df: pd.DataFrame, y: np.ndarray, pred: np.ndarray) -> dict:
    """Condition-head numbers on REAL test rows; every column of the comparison is computed on these same rows."""
    h = {}
    src = df["source"].to_numpy()
    fair_bdd = (src == "bdd") & (df["in_v2_train"].to_numpy() == 0)
    for name, mask in (("acdc_test", src == "acdc"), ("bdd_test_unseen_by_v2", fair_bdd), ("dawn_external", src == "dawn")):
        if mask.any():
            m = wd.classification_metrics(y[mask], pred[mask])
            h[f"{name}_accuracy"], h[f"{name}_macro_f1"] = m["accuracy"], m["macro_f1"]
    for cls in ("clear", "night", "rain", "snow"):
        m = fair_bdd & (y == wd.CLASS_TO_IDX[cls])
        if m.any():
            h[f"bdd_{cls}_recall"] = float((pred[m] == wd.CLASS_TO_IDX[cls]).mean())
    for cls in ("fog", "rain", "snow"):
        m = (src == "dawn") & (y == wd.CLASS_TO_IDX[cls])
        if m.any():
            h[f"dawn_{cls}_recall"] = float((pred[m] == wd.CLASS_TO_IDX[cls]).mean())
    m = src == "clean_cs"
    if m.any():
        h["cityscapes_clear_called_clear"] = float((pred[m] == wd.CLASS_TO_IDX["clear"]).mean())
    for beta in L.EVAL_BETAS:
        m = (src == "foggy_cs") & (df["extra"].to_numpy() == beta)
        if m.any():
            h[f"cityscapes_fog_beta_{beta}_called_fog"] = float((pred[m] == wd.CLASS_TO_IDX["fog"]).mean())
    m = src == "rain_cs"
    if m.any():
        h["cityscapes_rain_called_rain"] = float((pred[m] == wd.CLASS_TO_IDX["rain"]).mean())
    return h


def cmd_eval(args, device=None) -> None:
    device = device or v1.setup_device(args.cpu)
    df = get_manifest(args)
    ev = df[df["split"].isin(["val", "test"])].reset_index(drop=True)
    ensure_cache(ev, args)
    ensure_depth(ev)
    ckpts = find_ckpts(args)
    if not ckpts:
        sys.exit(f"[eval] no checkpoint found in {args.out} - run train first")
    _, w = v2.cap_workers(0, args.eval_workers, "eval")
    tta = not args.no_tta
    singles, sand_paths, sand_probs = {}, L.dawn_sand_images(args.root), []
    if args.limit_per_class:
        sand_paths = sand_paths[:12]
    for ck in ckpts:
        model, meta = L.load_model(ck, device)
        name = meta["config_name"] or ck.stem
        singles[name] = infer(model, ev, meta["config"], args, device, meta["fog_params"], w, f"eval {name}", tta=tta)
        if sand_paths:
            sand_probs.append(infer_files(model, meta, sand_paths, args, w, f"sandstorm probe {name}", tta)[0])
        del model
        free(device)
    first = next(iter(singles.values()))
    assert all(np.array_equal(o["y"], first["y"]) and np.array_equal(o["yf"], first["yf"]) for o in singles.values()), "models saw different rows"
    out = {k: first[k] for k in ("y", "yf", "yr")}
    for k in ("pc", "pf", "pr"):
        out[k] = np.mean([o[k] for o in singles.values()], axis=0)
    label = "ensemble of " + " + ".join(singles) if len(singles) > 1 else next(iter(singles))
    print(f"[eval] model: {label} | mirrored-image averaging: {tta}")

    pred, pf, pr = out["pc"].argmax(1), out["pf"].argmax(1), out["pr"].argmax(1)
    table = ev.copy()
    table["pred"], table["pred_fog_level"], table["pred_rain_level"] = [CLASSES[i] for i in pred], [L.FOG_LEVELS[i] for i in pf], [L.RAIN_LEVELS[i] for i in pr]
    for i, c in enumerate(CLASSES):
        table[f"p_{c}"] = out["pc"][:, i]
    table.to_csv(args.out / "eval_predictions.csv", index=False)

    metrics = {"model": label, "checkpoints": [str(c) for c in ckpts], "tta": tta, "per_source": {}}
    for split in ("val", "test"):
        metrics["per_source"][split] = {}
        for src in sorted(ev.loc[ev["split"] == split, "source"].unique()):
            m = ((ev["split"] == split) & (ev["source"] == src)).to_numpy()
            r = wd.classification_metrics(out["y"][m], pred[m])
            metrics["per_source"][split][src] = r
            print(f"[eval] {split}/{src}: n={r['n']} condition accuracy={r['accuracy']:.4f}")
    test = (ev["split"] == "test").to_numpy()
    t = ev[test].reset_index(drop=True)
    treal = (t["render"] == "").to_numpy()
    sub = {k: v[test] for k, v in out.items()}

    def levels(mask, which):
        y, p, names = (sub["yf"], sub["pf"].argmax(1), L.FOG_LEVELS) if which == "fog" else (sub["yr"], sub["pr"].argmax(1), L.RAIN_LEVELS)
        return level_metrics(np.where(mask, y, -1), p, names)

    cs = (t["domain"] == "cityscapes").to_numpy() & treal
    metrics["fog_level"] = {"cityscapes_real_dbf_test_cities": levels(cs, "fog"),
                            **{f"rendered_on_{s[:-5]}": levels((t["source"] == s).to_numpy(), "fog") for s in sorted(t["source"].unique()) if s.endswith("_rfog")}}
    metrics["rain_level"] = {"cityscapes_raincityscapes_test_cities": levels(cs, "rain"),
                             **{f"rendered_on_{s[:-6]}": levels((t["source"] == s).to_numpy(), "rain") for s in sorted(t["source"].unique()) if s.endswith("_rrain")}}
    sanity = {}
    for src, cls, key, names in (("acdc", "fog", "pf", L.FOG_LEVELS), ("bdd", "fog", "pf", L.FOG_LEVELS), ("dawn", "fog", "pf", L.FOG_LEVELS),
                                 ("acdc", "rain", "pr", L.RAIN_LEVELS), ("bdd", "rain", "pr", L.RAIN_LEVELS), ("dawn", "rain", "pr", L.RAIN_LEVELS)):
        m = ((t["source"] == src) & (t["label"] == cls)).to_numpy()
        if m.any():
            lv = sub[key][m].argmax(1)
            sanity[f"{src}_real_{cls}_predicted_{cls}_level"] = {n: int((lv == i).sum()) for i, n in enumerate(names)}
    metrics["real_weather_level_distribution_no_ground_truth"] = sanity
    for key in ("fog_level", "rain_level"):
        for name, m in metrics[key].items():
            if m.get("n"):
                print(f"[eval] {key.replace('_', ' '):10s} {name}: n={m['n']} accuracy={m['accuracy']:.4f} balanced={m['balanced_accuracy']:.4f} within-one={m['within_one_level']:.4f}")
    print("[eval] predicted severity on REAL fog / rain (no ground truth exists; shown for plausibility only):\n        " + json.dumps(sanity))
    if sand_probs:
        sp = np.mean(sand_probs, axis=0)
        metrics["dawn_sandstorm_out_of_distribution_probe"] = {
            "n": len(sand_paths), "predicted": {c: int((sp.argmax(1) == i).sum()) for i, c in enumerate(CLASSES)},
            "mean_top_probability": float(sp.max(1).mean()),
            "note": "sandstorm is a weather the model was never taught and is not a class; fog is the sensible answer"}
        print(f"[eval] out-of-distribution probe, {len(sand_paths)} DAWN SANDSTORM images (never taught, not a class): predicted "
              f"{metrics['dawn_sandstorm_out_of_distribution_probe']['predicted']} | mean confidence {sp.max(1).mean():.3f}")

    real_rows = t[treal].reset_index(drop=True)
    ty = sub["y"][treal]
    n_bdd, n_seen = int((real_rows["source"] == "bdd").sum()), int(((real_rows["source"] == "bdd") & (real_rows["in_v2_train"] == 1)).sum())
    if n_bdd:
        print(f"[eval] BDD100K test: {n_bdd} images, of which v2 was trained on {n_seen}; those are excluded from every bdd_* row below")
    comparison = {}
    if args.v2_ckpt.exists():
        y2, p2, _ = v2.predict_rows(args.v2_ckpt, real_rows, args, device, w, "eval v2")
        comparison["v2_retrained"] = headline(real_rows, y2, p2.argmax(1))
    if args.old_v3_ckpt.exists() and args.old_v3_ckpt not in ckpts:
        model, meta = L.load_model(args.old_v3_ckpt, device)
        o = infer(model, real_rows, meta["config"], args, device, meta["fog_params"], w, "eval v3")
        comparison["v3"] = headline(real_rows, o["y"], o["pc"].argmax(1))
        del model
        free(device)
    if len(singles) > 1:
        for name, o in singles.items():
            comparison[name] = headline(real_rows, ty, o["pc"][test][treal].argmax(1))
    comparison["v4_final"] = headline(real_rows, ty, sub["pc"][treal].argmax(1))
    cmp_table = pd.DataFrame(comparison)
    cmp_table.to_csv(args.out / "comparison.csv", float_format="%.4f")
    metrics["comparison_on_identical_real_test_rows"] = comparison
    print("[eval] condition head on identical REAL test rows (v4_final = " + label + "):\n" + cmp_table.to_string(float_format=lambda x: f"{x:.4f}"))

    for name, mask in (("all_real", treal), ("bdd", treal & (t["source"] == "bdd").to_numpy()), ("dawn", treal & (t["source"] == "dawn").to_numpy())):
        if mask.any():
            r = wd.classification_metrics(sub["y"][mask], sub["pc"][mask].argmax(1))
            wd.plot_confusion_matrix(r["confusion_matrix"], CLASSES, f"v4 condition, test/{name}: n={r['n']} acc={r['accuracy']:.3f}", args.out / f"confusion_condition_{name}.png")
    for key, names, fname in (("fog_level", L.FOG_LEVELS, "confusion_fog_level_cityscapes_test.png"), ("rain_level", L.RAIN_LEVELS, "confusion_rain_level_cityscapes_test.png")):
        m = next(iter(metrics[key].values()))
        if m.get("n"):
            wd.plot_confusion_matrix(m["confusion_matrix"], names, f"v4 {key.replace('_', ' ')}, Cityscapes test cities: n={m['n']} acc={m['accuracy']:.3f}", args.out / fname)
    L.save_json(metrics, args.out / "metrics.json")
    print(f"[eval] wrote metrics.json, comparison.csv, confusion_*.png, eval_predictions.csv -> {args.out}")


def cmd_predict(args) -> None:
    device = v1.setup_device(args.cpu)
    ckpts = find_ckpts(args)
    if not ckpts:
        sys.exit(f"[predict] no checkpoint found in {args.out} - run train first")
    pdir = args.predict_dir if args.predict_dir.is_absolute() else (args.root / args.predict_dir if (args.root / args.predict_dir).exists() else args.predict_dir.resolve())
    paths = wd.list_images(pdir)[: args.limit]
    if not paths:
        sys.exit(f"[predict] no images under {pdir}")
    _, w = v2.cap_workers(0, args.eval_workers, "predict")
    sums = None
    for ck in ckpts:
        model, meta = L.load_model(ck, device)
        probs = infer_files(model, meta, paths, args, w, f"predict {meta['config_name']}", not args.no_tta)
        sums = probs if sums is None else [a + b for a, b in zip(sums, probs)]
        del model
        free(device)
    pc, pf, pr = (s / len(ckpts) for s in sums)
    weather = [CLASSES[i] for i in pc.argmax(1)]
    table = pd.DataFrame({"frame_id": [p.stem for p in paths], "weather_pred": weather,
                          "fog_level": [L.level_name(x, L.FOG_LEVELS, wth == "fog") for x, wth in zip(pf, weather)],
                          "rain_level": [L.level_name(x, L.RAIN_LEVELS, wth == "rain") for x, wth in zip(pr, weather)]})
    for i, c in enumerate(CLASSES):
        table[f"p_{c}"] = pc[:, i]
    for i, c in enumerate(L.FOG_LEVELS):
        table[f"pfog_{c}"] = pf[:, i]
    for i, c in enumerate(L.RAIN_LEVELS):
        table[f"prain_{c}"] = pr[:, i]
    args.out.mkdir(parents=True, exist_ok=True)
    csv = args.out / f"predictions_{pdir.name}.csv"
    table.to_csv(csv, index=False, float_format="%.5f")
    print(f"[predict] {len(paths)} images, {len(ckpts)} model(s) -> {csv}\n[predict] weather {table['weather_pred'].value_counts().to_dict()} | fog levels "
          f"{table['fog_level'].value_counts().to_dict()} | rain levels {table['rain_level'].value_counts().to_dict()}")


def main(argv=None) -> None:
    args = parse_args(argv)
    wd.start_log(args.out / f"log_{args.cmd}.txt")
    {"prepare": cmd_prepare, "depth": cmd_depth, "lodo": cmd_lodo, "train": cmd_train, "eval": cmd_eval, "predict": cmd_predict}[args.cmd](args)


if __name__ == "__main__":
    main()
