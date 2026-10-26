# Core pipeline: detection sweep, trust model, danger score, runtime app

This package is the graded core of the project. It takes the finished weather model (v4, `src_v4/`) as an input
and implements everything in `PROJECT_GUIDE.md` section 7: ground-truth loaders, the per-frame record, the
detection sweep with the detector as a flag, failure labels, the four trust-feature families, the clear-only
trust model, the enhancement ON/OFF experiment, the danger gap, the quadrant, and the runtime application with a
**video-file mode** and a **webcam mode**.

```
frame -> weather + fog level (v4) -> enhancement (ON/OFF, per weather) -> frozen YOLO (v8m / 11m / 26m)
      -> flip-consistency -> trust features (quality on the RAW frame) -> XGBoost trust -> danger -> quadrant
```

## Files

| File | What it does |
|---|---|
| `common.py` | class vocabulary and COCO/BDD/ACDC/Cityscapes mapping, IoU matching, failure labels, AP50, the per-frame record, JSONL I/O |
| `gt_loaders.py` | BDD100K JSON, ACDC COCO (with the rain-file quirk), Cityscapes instance masks to boxes; seeded strata |
| `enhance.py` | luminance-only recipes keyed on (weather, fog level): CLAHE, dark-channel dehazing with guided filter, adaptive gamma, bilateral |
| `features.py` | (a) confidence statistics, (b) flip-consistency, (c) DINOv2 kNN distance to clear-day frames, (d) raw-frame quality, (e) weather probabilities |
| `detect.py` | Ultralytics wrapper; `yolov8m`, `yolo11m`, `yolo26m` (NMS-free) as a flag; weights cached in `weights/` |
| `danger.py` | danger formula, R_est vs R_true gap, quadrant, weight-sensitivity test |
| `weather.py` | batched v4 ensemble inference |
| `sweep.py` | `strata`, `detect`, `compare` commands |
| `trust.py` | `train` (clear-only) and `apply`; AUROC, risk-coverage, family ablation |
| `analysis.py` | danger gap, sensitivity, quadrant, severity sweep, enhancement ON vs OFF, figures |
| `pipeline.py` | the runtime `Pipeline` class used by the app and the benchmark |
| `app.py` | the application: `video` and `camera` modes |
| `bench.py` | latency / throughput / memory benchmark |
| `run_experiments.py` | the whole chain in order, resumable |

## Run the application

```powershell
conda activate weather
cd "<project root>"

# a video file on disk -> annotated mp4 + per-frame CSV (+ optional JSONL with the detections)
python src_core\app.py video --input "C:\path\to\drive.mp4" --output results\core\drive_out.mp4 --csv results\core\drive.csv --show

# the laptop webcam (press q or Esc in the window to stop)
python src_core\app.py camera --output results\core\cam.mp4 --csv results\core\cam.csv
```

Useful flags: `--no-enhance` (enhancement OFF), `--detector yolo11m`, `--conf 0.25`, `--weather-every 5`,
`--flip-every 2` (both trade accuracy of the slow parts for speed), `--no-dino` (faster, drops one trust feature
family), `--stride 2` (video: every other frame), `--max-frames N`, `--device cpu`.

The overlay shows weather and fog level, whether enhancement fired, the detections, trust and danger with bars,
the quadrant, and a red warning for **low trust with low apparent danger**: the corner where the scene looks safe
because perception may be failing.

Without a trained trust model (`results/core/trust/trust_model.json`) the app still runs and shows trust as n/a.

## Run the experiments

```powershell
python src_core\run_experiments.py                     # everything: ~1-2 h on the RTX 5060
python src_core\run_experiments.py --limit 20          # 15-minute smoke run of the whole chain
```

Or step by step:

```powershell
python src_core\sweep.py strata                                   # seeded frame lists -> results/core/strata.json
python src_core\sweep.py detect --model yolo26m                   # records + per-stratum metrics (enhancement off)
python src_core\sweep.py detect --model yolo11m
python src_core\sweep.py detect --model yolov8m
python src_core\sweep.py compare                                  # chooses the detector -> detector_choice.json
python src_core\sweep.py detect --model yolo26m --synthetic       # + imagecorruptions fog/frost/blur at severity 1,3,5
python src_core\sweep.py detect --model yolo26m --enhancement on
python src_core\trust.py train --records results\core\records\yolo26m_off.jsonl
python src_core\analysis.py --records results\core\records\yolo26m_off.jsonl --records-on results\core\records\yolo26m_on.jsonl
python src_core\bench.py --frames 300
```

## Strata (default caps, seed 0)

| Stratum | Frames | Camera vs trust training | Purpose |
|---|---|---|---|
| bdd_train_clear | 3,000 | same | trust-model TRAINING set and kNN reference |
| bdd_val_clear / night | 1,500 each | same | held-out same-camera clear and night |
| bdd_val_rain / snow | 396 / 422 (all) | same | held-out same-camera rain and snow |
| acdc_fog / night / rain / snow | 500 each | different | real adverse weather, GoPro |
| cs_clear + cs_fog_light / medium / dense | 800 scenes x 4 | different | identical scenes at 4 fog levels: camera-only baseline + weather effect |
| syn_fog / frost / motion_blur x {1,3,5} | 200 each | same | controlled severity sweep |

Detections are stored down to confidence 0.05; all metrics use the operating threshold 0.25 and ignore boxes
under 12 px high. Failure label = frame F1 < 0.5 OR any missed pedestrian (`fail`); `fail_person` and `fail_f1`
are stored too.

## Outputs (`results/core/`)

`strata.json`, `strata_counts.csv`, `records/<model>_<on|off>.jsonl`, `metrics_<model>_<on|off>.csv`,
`detector_comparison_off.csv`, `detector_choice.json`, `knn_reference_bdd_train_clear.npy`,
`trust/{trust_model.json, trust_model.meta.json, trust_results.csv, trust_ablation.csv, feature_importance.csv,
trust_scores.csv, risk_coverage.png, trust_auroc.png}`, `danger_gap.csv`, `danger_weight_sensitivity.csv`,
`quadrant_counts.csv`, `enhancement_on_vs_off.csv`, `figures/*.png`, `benchmark.json`, `summary.json`, `log_*.txt`.

## Tests

```powershell
python -m pytest tests -q                 # fast: 36 tests, no models (about 3 s)
python -m pytest tests -q --run-slow      # + the real-model integration test (GPU, datasets)
```
