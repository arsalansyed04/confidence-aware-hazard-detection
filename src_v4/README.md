# Weather classifier v4: condition + fog level + rain level (2-model ensemble)

A separate experiment. Nothing in `src/`, `src_cityscapes/`, `src_v3/` or their results is modified; v4 reuses
`src/` and `src_cityscapes/` as libraries. Outputs go to `results/weather_v4/`. v3 (trained before the full BDD100K
images existed) stays frozen in `src_v3/` + `results/weather_v3/` as the fallback and appears as the `v3`
comparison column.

## What v4 predicts

| Head | Classes |
|---|---|
| condition | clear, fog, night, rain, snow |
| fog level | none, light, medium, dense (visibility roughly over 1 km, 600 m, 300 m, 150 m) |
| rain level | none, light, medium, heavy; reported as `unrated` when the head has no opinion (real rain, see limits) |

Fog levels follow the Foggy Cityscapes attenuation coefficients (beta 0.005 / 0.01 / 0.02; visibility = 2.996 / beta).

## Why it is built this way (sources)

* Fog-density estimation trained on clean + Foggy Cityscapes beta levels and applied to real fog: Sakaridis et al.,
  "Model Adaptation with Synthetic and Real Data for Semantic Dense Foggy Scene Understanding" (ECCV 2018) and the
  CMAda follow-up (IJCV 2020).
* Severity labels exist for one camera only. v2 showed where that leads (a model that recognises the camera), so
  v4 renders fog and rain onto clear images of ALL cameras with `I = J*t + A*(1-t)`, `t = exp(-beta*d)` and an
  estimated depth map (Depth Anything V2 small), as in recent multi-class fog-density work (arXiv 2608.01572).
  The renderer is calibrated against the REAL Foggy Cityscapes-DBF frame of the same scene at all three fog levels,
  with a penalty on signed bias; error and bias per level are stored in `fog_render_params.json`.
* Recipe choices follow "Bridging the Generalization Gap in Adverse Weather Segmentation: A Training Recipe
  Perspective" (arXiv 2605.27962): lightly tuned pretrained backbones, scene-balanced sampling, moderate synthetic
  degradation, no restoration preprocessing. Recipes are BENCHMARKED, not assumed.
* "Does not overfit" is measured: `lodo` trains on two cameras and tests on the third, for every camera and recipe.
* Final model = average of the best recipe and the best recipe with a DIFFERENT backbone, plus mirrored-image
  averaging. The first benchmark showed DINOv2 stronger on condition and ConvNeXt stronger on fog level.

## Data roles

| Data | train | val | test |
|---|---|---|---|
| ACDC | official train | official val | official test |
| BDD100K (full 100k images) | train labels, 95% of videos, all usable images (about 54,800) | train labels, 5% of videos | official val labels, 8,193 images |
| Cityscapes clean / fog / rain | 16 cities | erfurt, weimar | frankfurt, lindau, munster |
| Rendered fog / rain on clear frames | follows the base image's split | same | same, fixed severity |
| DAWN fog / rain / snow (704) | never | never | external test, unknown cameras |
| DAWN sandstorm (323) | never | never | out-of-distribution probe only; not a class |

Sandstorm is deliberately NOT a class: DAWN is its only source (a class defined by one small web dataset teaches the
dataset's look, the v2 shortcut again), and training on any part of DAWN would destroy the only unseen-camera test.

BDD100K test images that v2 was trained on (318 of 8,193) are excluded from every `bdd_*` comparison row.

## Run (Anaconda PowerShell Prompt, project root)

```powershell
conda activate weather
cd "<project root>"
python src_v4\train_weather_v4.py prepare
python src_v4\train_weather_v4.py depth
python src_v4\train_weather_v4.py lodo
python src_v4\train_weather_v4.py train
python src_v4\train_weather_v4.py predict --predict-dir bdd100k-10k\val
```

`prepare` and `depth` are incremental (only new images are processed). `lodo` is optional: without it `train` uses
dinov2_last2 + convnext_full, the two best recipes of the first benchmark. Loader processes are capped from free
memory automatically; `--workers 0 --eval-workers 0` is the always-safe fallback.

## What the results can and cannot show

* Fog-level and rain-level accuracy are measured on real Foggy / Rain Cityscapes test cities and on rendered
  fog / rain on held-out images of every camera.
* No dataset labels the severity of REAL fog or REAL rain, so that cannot be scored; `eval` prints the predicted
  level distribution on real fog and rain for plausibility only.
* Measured 19 Sept 2026: the rain-level head answers "none" on 99% of REAL rain, because rain severity could only be
  learned from synthetic rain. `predict` reports `unrated` there instead of inventing a level. More data cannot
  fix this: no real-rain severity labels exist.
* BDD100K weather tags are noisy (wet roads tagged rainy, snow on the ground under a clear sky tagged snowy), so
  BDD100K accuracy has a ceiling below 100% for any model.

## Outputs (`results/weather_v4/`)

`manifest.csv`, `counts_real.csv`, `counts_rendered.csv`, `fog_render_params.json`, `lodo_results.csv`,
`lodo_summary.csv`, `lodo_best.json`, `ckpt_<recipe>.pt` (the ensemble members), `history_<recipe>.csv`,
`sampling_shares.csv`, `metrics.json`, `comparison.csv` (v2, v3, each member, final v4 ensemble on identical
real test rows), `confusion_condition_{all_real,bdd,dawn}.png`, `confusion_fog_level_cityscapes_test.png`,
`confusion_rain_level_cityscapes_test.png`, `eval_predictions.csv`, `predictions_<folder>.csv`, `log_<command>.txt`.

## Use from the pipeline

```python
import sys, glob; sys.path.insert(0, "src_v4")
from weather_v4_lib import load_ensemble, predict_image_ensemble
models = load_ensemble(sorted(glob.glob("results/weather_v4/ckpt_*.pt")), "cuda")
r = predict_image_ensemble(models, bgr_frame)   # r["weather"], r["fog_level"], r["rain_level"], r["p_fog_level"], ...
```
