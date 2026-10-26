# Core results (measured 12-13 October 2026, RTX 5060 Laptop)

All numbers come from the files in this folder. Detector YOLO26m unless stated; enhancement OFF unless stated;
operating confidence 0.25; boxes under 12 px high ignored; failure = frame F1 < 0.5 or any missed pedestrian.
Strata: 12,024 real frames (section 2) + 1,800 synthetic-corruption frames.

## 1. Detector choice (`detector_comparison_off.csv`, `detector_choice.json`)

| adverse strata (night, rain, snow, fog of every camera) | YOLO26m | YOLO11m | YOLOv8m |
|---|---|---|---|
| pedestrian recall | 0.443 | 0.409 | 0.413 |
| precision | 0.688 | 0.714 | 0.681 |
| mAP50 | 0.409 | 0.385 | 0.379 |
| pedestrian recall, clear strata | 0.477 | 0.446 | 0.459 |
| drop clear to adverse | 0.034 | 0.037 | 0.046 |

Rule: best adverse pedestrian recall among models within 5 precision points of the best; tie -> smallest drop.
**Chosen: YOLO26m.** It also needs no NMS threshold.

## 2. Detection per stratum, YOLO26m, enhancement OFF (`metrics_yolo26m_off.csv`)

| stratum | frames | mAP50 | pedestrian recall | precision | frames with a missed pedestrian | weather model agrees with GT |
|---|---|---|---|---|---|---|
| bdd_train_clear (trust training) | 3000 | 0.466 | 0.482 | 0.810 | 0.279 | 0.977 |
| bdd_val_clear | 1500 | 0.475 | 0.493 | 0.808 | 0.286 | 0.965 |
| bdd_val_night | 1500 | 0.335 | 0.352 | 0.707 | 0.182 | 0.993 |
| bdd_val_rain | 396 | 0.493 | 0.468 | 0.803 | 0.283 | 0.801 |
| bdd_val_snow | 422 | 0.441 | 0.542 | 0.800 | 0.296 | 0.791 |
| acdc_fog | 500 | 0.545 | 0.559 | 0.613 | 0.148 | 1.000 |
| acdc_night | 506 | 0.244 | 0.322 | 0.553 | 0.547 | 1.000 |
| acdc_rain | 500 | 0.355 | 0.349 | 0.572 | 0.278 | 1.000 |
| acdc_snow | 500 | 0.428 | 0.553 | 0.592 | 0.418 | 0.998 |
| cs_clear | 800 | 0.457 | 0.461 | 0.741 | 0.671 | 0.998 |
| cs_fog_light | 800 | 0.440 | 0.450 | 0.741 | 0.676 | 0.999 |
| cs_fog_medium | 800 | 0.425 | 0.433 | 0.744 | 0.682 | 1.000 |
| cs_fog_dense | 800 | 0.387 | 0.398 | 0.752 | 0.705 | 1.000 |

The four Cityscapes rows are the SAME 800 scenes with the same ground truth: the camera-only baseline is the
`cs_clear` row, and the rise from it is the weather effect alone (mAP50 0.457 -> 0.387 at dense fog). Cityscapes'
high "missed pedestrian" rates come from its dense annotation of small and occluded people.

Synthetic severity sweep on 200 BDD100K clear frames (`imagecorruptions`):

| corruption | severity 1 | 3 | 5 |
|---|---|---|---|
| fog: mAP50 / ped. recall | 0.477 / 0.438 | 0.453 / 0.423 | 0.447 / 0.408 |
| frost: mAP50 / ped. recall | 0.447 / 0.375 | 0.294 / 0.292 | 0.275 / 0.257 |
| motion blur: mAP50 / ped. recall | 0.448 / 0.370 | 0.264 / 0.179 | 0.144 / 0.088 |

## 3. Trust model (`trust/`)

XGBoost on 30 features (quality, confidence, flip-consistency, DINOv2 distance, weather probabilities), trained on
the 3,000 clear-day BDD100K frames ONLY (failure rate 0.355). AUROC of predicting failure; `conf only` = the
mean detection confidence baseline. `fail@70` = residual failure rate when acting on the most-trusted 70%.

| stratum | failure rate | AUROC trust | AUROC conf only | fail@100 | fail@70 | fail@50 |
|---|---|---|---|---|---|---|
| bdd_train_clear (in-sample, ceiling only) | 0.355 | 0.986 | 0.622 | 0.355 | 0.094 | 0.003 |
| bdd_val_clear | 0.368 | 0.761 | 0.654 | 0.368 | 0.245 | 0.197 |
| bdd_val_night | 0.571 | 0.730 | 0.651 | 0.571 | 0.470 | 0.409 |
| bdd_val_rain | 0.359 | 0.750 | 0.592 | 0.359 | 0.235 | 0.207 |
| bdd_val_snow | 0.370 | 0.729 | 0.619 | 0.370 | 0.264 | 0.227 |
| acdc_fog | 0.316 | 0.644 | 0.603 | 0.316 | 0.257 | 0.228 |
| acdc_night | 0.721 | 0.637 | 0.544 | 0.721 | 0.678 | 0.652 |
| acdc_rain | 0.498 | 0.663 | 0.612 | 0.498 | 0.431 | 0.380 |
| acdc_snow | 0.564 | 0.656 | 0.605 | 0.564 | 0.509 | 0.444 |
| cs_clear | 0.700 | 0.731 | 0.648 | 0.700 | 0.607 | 0.545 |
| cs_fog_light / medium / dense | 0.70 / 0.72 / 0.74 | 0.733 / 0.733 / 0.744 | 0.656 / 0.670 / 0.676 | | 0.625 / 0.639 / 0.664 | 0.562 / 0.585 / 0.598 |
| syn_fog 1 / 3 / 5 | 0.43 / 0.44 / 0.47 | 0.774 / 0.765 / 0.761 | 0.750 / 0.745 / 0.711 | | 0.286 / 0.329 / 0.314 | 0.250 / 0.280 / 0.260 |
| syn_frost 1 / 3 / 5 | 0.45 / 0.60 / 0.60 | 0.807 / 0.742 / 0.732 | 0.662 / 0.680 / 0.637 | | 0.300 / 0.486 / 0.486 | 0.250 / 0.420 / 0.420 |
| syn_motion_blur 1 / 3 / 5 | 0.46 / 0.59 / 0.82 | 0.728 / 0.680 / 0.684 | 0.682 / 0.603 / 0.569 | | 0.350 / 0.493 / 0.793 | 0.310 / 0.460 / 0.730 |

* The trust model beats the confidence-only baseline in **every one of the 21 held-out strata**.
* Transfer: same camera, unseen weather: 0.73-0.76. Different camera (ACDC): 0.64-0.66. Different camera,
  controlled fog (Cityscapes): 0.73-0.74, rising slightly with fog density.
* Risk-coverage example: on held-out clear day, acting on the trusted half of frames cuts the failure rate from
  36.8% to 19.7%; at night (BDD100K) from 57.1% to 40.9%.
* Family ablation (pooled AUROC over all held-out strata): full 0.739; without quality 0.727; without confidence
  0.690; without flip 0.731; without DINOv2 0.729; without weather probabilities 0.741. Confidence statistics
  are the most important family; the weather probabilities add nothing (the model is marginally better without
  them). Top single features: number of detected persons, person flip-consistency, class entropy, overall
  flip-consistency, max confidence.

## 4. Danger gap R_true - R_est (`danger_gap.csv`, `danger_weight_sensitivity.csv`, `quadrant_counts.csv`)

| stratum | R_true | R_est | gap | frames underestimated by > 0.1 |
|---|---|---|---|---|
| bdd_val_clear | 0.618 | 0.597 | 0.021 | 9.1% |
| bdd_val_night | 0.718 | 0.627 | **0.091** | **35.7%** |
| bdd_val_rain | 0.741 | 0.710 | 0.031 | 11.4% |
| bdd_val_snow | 0.770 | 0.748 | 0.022 | 9.2% |
| acdc_fog | 0.668 | 0.654 | 0.014 | 12.4% |
| acdc_night | 0.614 | 0.511 | **0.104** | **46.4%** |
| acdc_rain | 0.561 | 0.531 | 0.030 | 18.4% |
| acdc_snow | 0.686 | 0.661 | 0.025 | 15.2% |
| cs_clear -> light -> medium -> dense fog | 0.641 / 0.791 / 0.851 / 0.911 | 0.622 / 0.767 / 0.822 / 0.868 | 0.019 / 0.024 / 0.029 / 0.042 | 5.9 / 8.0 / 9.8 / 13.9% |
| syn_frost 1 / 3 / 5 | 0.614 | 0.569 / 0.518 / 0.496 | 0.045 / 0.096 / 0.117 | 16 / 30 / 40% |
| syn_motion_blur 1 / 3 / 5 | 0.614 | 0.598 / 0.568 / 0.520 | 0.016 / 0.046 / 0.094 | 9 / 21 / 39% |

* The gap is positive in every stratum: perception underestimates danger, most at night (missed pedestrians
  remove the highest-weight terms) and it grows monotonically with fog density and with synthetic severity.
* Weight sensitivity (each class weight +-50%, 2,000 frames): minimum Spearman 0.954, minimum high/medium/low
  tercile agreement 0.875. The ranking of scenes is stable.
* Quadrant counts (trust < 0.5, danger < 0.4 = "low trust, low apparent danger"): night 363 of 2,006 frames
  (18%), fog 98 of 2,900, clear 235 of 4,100 (6%).

## 5. Enhancement ON vs OFF, YOLO26m, identical frames (`enhancement_on_vs_off.csv`)

| stratum | recipe fired | d mAP50 | d pedestrian recall | d precision | d missed-pedestrian rate | d conf of true positives |
|---|---|---|---|---|---|---|
| clear (BDD, Cityscapes) | none (by design) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| bdd_val_night | gamma + CLAHE + bilateral | -0.007 | **+0.065** | -0.034 | -0.007 | -0.002 |
| acdc_night | same | +0.011 | **+0.054** | +0.014 | -0.016 | +0.012 |
| bdd_val_rain / acdc_rain | CLAHE | -0.004 / -0.010 | +0.007 / +0.030 | 0.000 / +0.013 | 0.000 / -0.006 | +0.002 / -0.001 |
| bdd_val_snow / acdc_snow | gamma(bright) + CLAHE | -0.001 / 0.000 | 0.000 / +0.009 | +0.003 / +0.002 | -0.002 / -0.004 | +0.002 / +0.004 |
| acdc_fog | per fog level | +0.007 | +0.015 | -0.008 | -0.004 | +0.002 |
| cs_fog_light / medium / dense | CLAHE / dehaze+CLAHE / dehaze+CLAHE+denoise | +0.008 / +0.020 / +0.028 | +0.001 / +0.020 / +0.042 | -0.005 / -0.009 / -0.021 | -0.006 / -0.014 / -0.024 | +0.001 / +0.001 / +0.001 |

* Night: +5 to +7 points of pedestrian recall, at a small precision cost on BDD100K. Dense fog: +4 points.
* Rain and snow: within noise, as predicted.
* No confidence inflation anywhere (true-positive confidence changes by at most +0.012): the enhancement does
  not fool the detector into being more confident without being more correct.
* The synthetic strata were not re-run with enhancement ON (NaN rows).

## 6. Runtime performance (`benchmark.json`, `benchmark_fast.json`), 1280x720 frames, trust model loaded

| configuration | sustained fps | latency p50 / p95 / p99 (ms) | GPU peak | RSS growth over 300 frames |
|---|---|---|---|---|
| default: weather every 5, flip every 2, DINOv2 on | 20.6 | 47 / 71 / 81 | 439 MB | +0.5 MB |
| fast: DINOv2 off, flip every 5 | 34.8 | 19 / 54 / 76 | 355 MB | -2.2 MB |

Video-file mode on a 30-frame 720p clip with everything on: 7.9 fps including file decoding and mp4 encoding.
Webcam mode headless at 640x480 with the GPU shared with a running sweep: 3.8 fps (10 frames).

## 7. Figures (`figures/`, `trust/`)

`trust/trust_auroc.png`, `trust/risk_coverage.png`, `figures/danger_gap.png`, `figures/quadrant.png`,
`figures/severity_sweep.png`, `figures/enhancement_on_vs_off.png`.
