# Defect Detector

Industrial anomaly detection on MVTec AD using a frozen OpenCLIP ViT-B/32 backbone.

## Detector architecture

The current detector combines:

- Frozen OpenCLIP ViT-B/32 image/text features.
- A compositional normal/anomalous prompt ensemble.
- Three final visual-transformer intermediate feature maps for local scoring.
- A category-specific normal-reference patch memory built only from MVTec `train/good` images.
- Robust top-10% local aggregation instead of a single maximum patch.
- Learned language/reference fusion weights and temperature scaling fitted on the separate calibration split.
- Image-level AUROC, pixel AU-PRO, and ECE reporting.

The reference memory is deliberately separated from calibration/evaluation data to avoid test leakage.

## Requirements

- Python 3.11+ recommended
- CPU-compatible PyTorch or a CUDA PyTorch installation
- Dependencies in `backend/requirements.txt`
- MVTec AD extracted to `data/mvtec_ad/`

## Setup

From the repository root:

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r backend/requirements.txt
```

Generate deterministic manifests if they do not already exist:

```bash
python backend/scripts/create_mvtec_manifests.py
```

## Refit the upgraded detector calibration

The committed calibration file is intentionally reset to temperature 1.0 because the previous calibration had an unusably large temperature. Refit it on the calibration split before reporting final probabilities:

```bash
python backend/scripts/fit_temperature.py --device cpu
```

For CUDA:

```bash
python backend/scripts/fit_temperature.py --device cuda
```

The first run builds a category-specific normal patch cache under `data/reference_cache/`. This can take time because the official MVTec train/good images are encoded once.

## Evaluate

After calibration:

```bash
python backend/scripts/evaluate_mvtec.py --device cpu
```

The report is written to:

```text
artifacts/evaluations/results.json
artifacts/evaluations/results.csv
```

For a quick category-specific run:

```bash
python backend/scripts/evaluate_mvtec.py --device cpu --categories bottle cable zipper
```

## Important evaluation note

The repository's manifests use a deterministic held-out subset of the MVTec test set for calibration/evaluation. The calibration images must never be used for fitting the normal-reference memory; only `train_reference.json` is used for that memory.

The repository's pixel AU-PRO implementation is an approximation. For publication/benchmark claims, validate the final anomaly maps with the official MVTec evaluation procedure.

## Tests

Run:

```bash
pytest backend/tests
```

## Repository layout

```text
backend/
  app/core/
    clip_encoder.py       # frozen CLIP + multi-layer patch features
    prompts.py            # compositional prompt ensemble
    reference_bank.py     # normal-reference patch memory
    fusion.py             # language/reference anomaly fusion
    calibration.py        # probability calibration
  scripts/
    create_mvtec_manifests.py
    fit_temperature.py
    evaluate_mvtec.py
data/
  manifests/
  mvtec_ad/               # local MVTec dataset
artifacts/
```
