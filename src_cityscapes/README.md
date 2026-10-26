# Weather classifier v2: + Foggy Cityscapes + RainCityscapes

A separate experiment. It never writes to `src/` or to the v1 model in `results/weather_classifier/`
(it only adds resized JPEGs to that folder's shared `cache/`). Outputs go to
`results/weather_classifier_cityscapes/`.

Note: RainCityscapes is excluded from the main project (handoff section 3). It is used here, for this
side experiment on the runtime weather classifier only.

## Splits

| Data | train | val | test |
|---|---|---|---|
| ACDC | official train | official val | official test |
| BDD100K (labelled) | all 2,486 | none | none |
| Foggy + Rain Cityscapes | 16 cities | erfurt, weimar | frankfurt, lindau, munster |

Cityscapes-family images are split **by city**, with one city list shared by fog and rain. The filename
parameters (fog `beta`; rain `alpha`, `beta`, `dropsize`, `pattern`) are renderings of the same street
frame, so a by-city split keeps all 3 fog levels and all 36 rain renderings of a scene in one split.
The test cities are Cityscapes' own official validation cities.

| Source | train images | train scenes |
|---|---|---|
| ACDC | 3,200 | 3,200 |
| BDD100K | 2,486 | 2,486 |
| Foggy Cityscapes | 8,172 | 2,724 |
| RainCityscapes | 7,272 | 202 |

Because rain has only 202 scenes drawn 36 ways, training samples **per scene, not per image**: each class gets
1/5 of an epoch's draws, each source's share inside a class is proportional to sqrt(scenes), each scene in a
source is equally likely, each rendering of a scene is equally likely. `sampling_shares.csv` records it.

## What the result can and cannot show

There are no clean Cityscapes images on disk, so every Cityscapes-camera image is fog or rain. The model can
therefore score well on the Cityscapes test cities partly by recognising the camera. After this training the
Foggy Cityscapes numbers are **in-domain** (unseen cities, seen camera and renderer); they are no longer the
out-of-domain probe they were for v1. Only `acdc_test_*` is directly comparable with v1's headline accuracy.
The clean package (`leftImg8bit_trainvaltest`, 11 GB) is now on disk: `prepare` adds its train and val frames
as `clear` with the same city split (its unlabelled `test/` folder is not used), which removes the shortcut.
`best_no_clean.pt` keeps the model trained before those frames existed; `eval` scores it as the
`v2_no_clean` column next to `v1` and the retrained `v2`, on identical test rows.

## Run (Anaconda PowerShell Prompt, project root)

```powershell
conda activate weather
cd "<project root>"
python src_cityscapes\train_weather_cityscapes.py prepare
python src_cityscapes\train_weather_cityscapes.py train
python src\clip_zeroshot_weather.py --out results\weather_classifier_cityscapes
python src\train_weather_classifier.py predict --out results\weather_classifier_cityscapes --predict-dir bdd100k-10k\val
```

Memory: loader processes are capped automatically from free RAM and free Windows commit at start
(`[memory]` line). Close the browser first for a faster run. `--workers 0 --eval-workers 0` is the
always-safe fallback.

## Outputs

`manifest.csv`, `counts_images.csv`, `counts_scenes.csv`, `sampling_shares.csv`, `best.pt`, `last.pt`,
`history.csv`, `training_curves.png`, `metrics.json` (per source, per fog beta, per rain intensity, with mean
class probability per level), `comparison_v1_vs_v2.csv` (v1 and v2 on identical test rows),
`confusion_matrix_test_{all,acdc,cityscapes}.png`, `eval_predictions.csv`, `log_<command>.txt`.
