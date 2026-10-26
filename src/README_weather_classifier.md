# Weather classifier (clear / fog / night / rain / snow)

Runtime-only component of the perception pipeline. It is evaluated against dataset
labels with a confusion matrix and is **never** used to stratify detection results
(stratification always uses the datasets' ground-truth weather tags).

| Item | Value |
|---|---|
| Model | torchvision `mobilenet_v3_large`, ImageNet-1K V2 weights, head replaced by `Linear(1280, 5)`, all layers fine-tuned |
| Train / val / test | ACDC official splits as shipped. Condition folder = label; every `*_ref` frame = `clear` |
| OOD probe | Foggy Cityscapes-DBF (3 fog levels), eval-only, never trained on |
| Optional | BDD100K rows are added automatically when a labels JSON exists in `bdd100k-10k/labels/` |
| Selection | best epoch by macro-F1 on ACDC val; ACDC test is touched once by `eval` |
| Augmentation | RandomResizedCrop(0.6-1.0) + horizontal flip only. No colour/brightness jitter (night and fog cues) |
| Loss / optimiser | class-weighted cross-entropy (label smoothing 0.05), AdamW, cosine LR with 1-epoch warm-up, bf16 autocast |

Files: `train_weather_classifier.py` (CLI + model), `weather_data.py` (manifest, cache, transforms, metrics),
`clip_zeroshot_weather.py` (CLIP zero-shot baseline on the same manifest).

## Run (Anaconda PowerShell Prompt, from the project root)

```powershell
conda activate weather
cd "<project root>"

# 0. smoke test (~2 min, separate --out so nothing real is overwritten)
python src\train_weather_classifier.py train --out results\smoke --limit-per-class 32 --epochs 1 --workers 2

# 1. build manifest + counts + resized cache once (~18k images)
python src\train_weather_classifier.py prepare

# 2. train 12 epochs; eval on best.pt runs automatically at the end
python src\train_weather_classifier.py train

# 3. (re-)evaluate any time
python src\train_weather_classifier.py eval

# 4. weather_pred for unlabelled frames (CSV: frame_id, weather_pred, p_clear, p_fog, p_night, p_rain, p_snow)
python src\train_weather_classifier.py predict --predict-dir bdd100k-10k\val

# 5. CLIP zero-shot baseline (first run downloads 605 MB of weights over plain HTTP; allow 15-20 min on a 5 Mbit/s link)
python src\clip_zeroshot_weather.py
```

`--help` on any subcommand lists every flag with its default. `--resume` continues from `last.pt`.

## Outputs (`results/weather_classifier/`)

`manifest.csv`, `counts.csv`, `manifest_info.json`, `cache/`, `best.pt`, `last.pt`, `history.csv`,
`training_curves.png`, `metrics.json`, `confusion_matrix_val.png`, `confusion_matrix_test.png`,
`foggy_cityscapes_probe.json`, `eval_predictions.csv`, `predictions_<folder>.csv`, `log_<command>.txt`,
`clip_zeroshot/metrics_clip.json` + confusion matrices.

## BDD100K labels (optional, adds a `bdd` source and a `bdd_val` metrics block)

Download `det_train.json` and `det_val.json` (Detection 2020 labels, Scalabel format with
`attributes.weather` / `attributes.timeofday`) and put them in `bdd100k-10k\labels\`.
The next `train` or `prepare` detects the new files and rebuilds the manifest automatically
(images are matched to `bdd100k-10k\train` and `\val` by filename; dawn/dusk and undefined rows are dropped).

## Use from the runtime pipeline

```python
import sys; sys.path.insert(0, "src")
from train_weather_classifier import load_model, predict_image
model, meta = load_model("results/weather_classifier/best.pt", "cuda")
label, probs = predict_image(model, meta, bgr_frame)   # bgr_frame: HxWx3 uint8 from cv2
```
