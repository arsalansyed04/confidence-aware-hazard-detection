# Weather model checkpoints

The final weather model is an ensemble of two checkpoints:

| file | backbone | size | in this folder |
|---|---|---|---|
| `ckpt_dinov2_last2.pt` | DINOv2 ViT-S/14, last 2 blocks tuned | 88 MB | yes |
| `ckpt_convnext_full.pt` | ConvNeXt-Tiny, fully tuned | 113 MB | no: over GitHub's 100 MB file limit |

The application runs with the one checkpoint present (a single-model prediction instead of the ensemble; the
DINOv2 member alone scores 99.68% on ACDC test against 99.72% for the ensemble, see `comparison.csv`). To restore the
ensemble, either copy `ckpt_convnext_full.pt` from the original results folder, store it with Git LFS, or retrain it:

    python src_v4/train_weather_v4.py train --configs convnext_full --no-final-eval

The trust model (`results/core/trust/trust_model.json`), its feature list, and the kNN reference of clear-day
embeddings (`results/core/knn_reference_bdd_train_clear.npy`) are included and are all the runtime needs.
