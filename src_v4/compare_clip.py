"""CLIP zero-shot vs v2 / v3 / v4 on the IDENTICAL real test images of the v4 manifest.

Reads results/weather_v4/manifest.csv and comparison.csv, runs open_clip ViT-B-32-quickgelu (OpenAI weights, the same
prompt ensemble as src/clip_zeroshot_weather.py) on every REAL test row (rendered rows are excluded: CLIP would be
shown the un-rendered clear image), scores it with the same headline() function as v4's table, and writes
results/weather_v4/comparison_with_clip.csv. Inference only.

    python src_v4/compare_clip.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_weather_v4 as T  # noqa: E402  (headline(), the v4 library and the v1 helpers)

sys.path.insert(0, str(T.L.PROJECT_ROOT / "src"))
import clip_zeroshot_weather as C  # noqa: E402  (prompts and CLIP normalisation)

wd = T.wd


def main() -> None:
    out = T.L.PROJECT_ROOT / "results" / "weather_v4"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = T.L.load_manifest(out / "manifest.csv")
    rows = df[(df["split"] == "test") & (df["render"] == "")].reset_index(drop=True)
    print(f"[clip] {len(rows)} real test images: " + str(rows["source"].value_counts().to_dict()))

    import open_clip
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai", device=device)
    tok = open_clip.get_tokenizer("ViT-B-32-quickgelu")
    model.eval()
    with torch.inference_mode():
        text = torch.stack([F.normalize(F.normalize(model.encode_text(tok(C.PROMPTS[c]).to(device)).float(), dim=-1).mean(0), dim=0)
                            for c in wd.CLASSES])
        scale = float(model.logit_scale.exp())
    y = rows["label"].map(wd.CLASS_TO_IDX).to_numpy(dtype=np.int64)
    ds = wd.CachedDataset(rows["cache_path"], y, wd.EvalTransform(224))
    norm, probs = C.ClipNorm(device), []
    with torch.inference_mode():
        for imgs, _ in tqdm(DataLoader(ds, batch_size=128, num_workers=0), desc="clip", unit="batch", mininterval=1.0):
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                f = model.encode_image(norm(imgs))
            probs.append(torch.softmax(scale * F.normalize(f.float(), dim=-1) @ text.T, dim=1).cpu().numpy())
    pred = np.concatenate(probs).argmax(1)
    clip = T.headline(rows, y, pred)

    table = pd.read_csv(out / "comparison.csv", index_col=0)
    table.insert(0, "clip_zero_shot", pd.Series(clip))
    table.to_csv(out / "comparison_with_clip.csv", float_format="%.4f")
    print("[clip] identical REAL test rows:\n" + table.to_string(float_format=lambda x: f"{x:.4f}"))
    for src in ("bdd", "dawn"):
        m = (rows["source"] == src).to_numpy()
        if src == "bdd":
            m = m & (rows["in_v2_train"] == 0).to_numpy()
        cm = wd.classification_metrics(y[m], pred[m])["confusion_matrix"]
        print(f"[clip] {src} confusion (rows true, cols pred) {wd.CLASSES}:")
        for c, r in zip(wd.CLASSES, cm):
            if sum(r):
                print(f"         {c:6s} {r}")
    print(f"[clip] wrote {out / 'comparison_with_clip.csv'}")


if __name__ == "__main__":
    main()
