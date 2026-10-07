# Defect Detector

Industrial anomaly detection on MVTec AD using a frozen OpenCLIP ViT-B/32 backbone.

## Phase 1 detector architecture

Phase 1 is intentionally a **pure zero-shot CLIP detector**. It combines:

- Frozen OpenCLIP ViT-B/32 image/text features.
- A compositional normal/anomalous prompt ensemble.
- Three final visual-transformer intermediate feature maps for local scoring.
- Robust top-10% local aggregation instead of a single maximum patch.
- Fixed calibration-selected language-fusion weights: **GLOBAL_WEIGHT=0.95** and **LOCAL_WEIGHT=0.05**.
- Temperature-and-intercept calibration fitted only on the held-out calibration split.
- 7x7 multi-layer CLIP anomaly heatmaps with horizontal-flip test-time
  augmentation for localization. The anomaly orientation is
  `normal similarity - anomalous similarity`, selected on the calibration split.
- MVTec-compatible AU-PRO, pixel AUROC, image AUROC, and ECE reporting.

The category-specific normal-reference memory used in earlier experiments is **not part of Phase 1**. Its implementation is retained under `backend/experimental/` only as a decision trace; its large tensor cache was removed.

### Working principle

> The system asks CLIP whether an image looks semantically defective, both globally and in local patches, and calibrates that score into a probability.

## Active output

The evaluator records `defect_logit`, the primary calibrated
`defect_probability`, and a heatmap. The calibrated probability is:

```text
sigmoid(defect_logit / temperature + bias)
```

The current Phase 1 evaluation does not use the repository's experimental MC
Dropout scoring head.

## Requirements

- Python 3.12 recommended.
- CPU-compatible PyTorch or a CUDA PyTorch installation.
- Dependencies in `backend/requirements.txt`.
- MVTec AD extracted to `data/mvtec_ad/`.

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

Generate deterministic calibration/evaluation manifests if they do not already exist:

```bash
python backend/scripts/create_mvtec_manifests.py
```

The active Phase 1 manifest generator does not create a normal-reference cache.

## Fit calibration

Fit the positive temperature and intercept against the language-only anomaly score:

```bash
python backend/scripts/fit_temperature.py --device cpu
```

For CUDA:

```bash
python backend/scripts/fit_temperature.py --device cuda
```

The script writes a fresh `backend/config/calibration.json`. Do not hand-edit the fitted temperature.

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

Evaluation reports image AUROC, pixel AUROC, AU-PRO at FPR=0.30, and ECE.
Pixel metrics use heatmaps and masks downsampled to a maximum side of 256 by
default, which avoids multi-gigabyte allocations when evaluating a category.
Use `--metric-max-side 0` only when enough RAM is available for native-resolution
metrics.

### Heatmap polarity diagnostic

Heatmap polarity was selected on the held-out calibration split before the
evaluation split was touched: `inverted` (normal-prompt similarity minus
anomalous-prompt similarity) won with macro AU-PRO `0.4468` and macro pixel
AUROC `0.6884`, versus `0.0847` and `0.3116` for the native orientation. It is
therefore the default for evaluation. To reproduce the selection procedure on a
fresh calibration split, run native and inverted maps into separate directories:

```bash
python backend/scripts/evaluate_mvtec.py --device cpu --split-manifest data/manifests/calibration.json --categories bottle --heatmap-polarity native --output-dir artifacts/calibration-native
python backend/scripts/evaluate_mvtec.py --device cpu --split-manifest data/manifests/calibration.json --categories bottle --heatmap-polarity inverted --output-dir artifacts/calibration-inverted
```

Use the selected polarity unchanged on the evaluation split. The standard full
evaluation defaults to `--heatmap-polarity inverted` and downsampled metric maps
(`--metric-max-side 256`) so it can complete within typical workstation memory.

## Localization metric

Phase 1 uses the released MVTec-style PRO protocol: connected ground-truth regions are weighted equally, normal pixels determine FPR, and the PRO curve is integrated only up to FPR=0.30. The implementation uses exact score thresholds rather than the previous 200-threshold approximation, at the configured metric-map resolution.

The MVTec project page provides the reference evaluation code used for consistent benchmark evaluation. For publication claims, cross-check final maps/numbers against that released evaluator.

## Calibration sanity check

Before trusting ECE, inspect the fitted temperature and generate a reliability diagram from the evaluation probabilities. The calibration split is separate from the evaluation split, and no evaluation labels are used for fitting.

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
    fusion.py             # pure zero-shot language fusion
    calibration.py        # temperature calibration
    mc_dropout.py         # uncertainty estimation
    evaluation.py         # AUROC, ECE, validated AU-PRO
    heatmap.py            # heatmap visualization
  experimental/
    README.md             # trace of excluded reference-memory experiment
  scripts/
    create_mvtec_manifests.py
    fit_temperature.py
    evaluate_mvtec.py
data/
  manifests/
  mvtec_ad/               # local MVTec dataset
artifacts/
```

## Phase 1 scope

The following files are deliberately unchanged by the Phase 1 detector redesign:

- `backend/app/core/clip_encoder.py`
- `backend/app/core/prompts.py`
- `backend/app/core/heatmap.py`

The objective is to finish a clean zero-shot CLIP baseline before adding any later architecture.
