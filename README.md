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
- MC Dropout uncertainty metadata from the existing lightweight scoring head.
- 7x7 multi-layer CLIP anomaly heatmaps with horizontal-flip test-time
  augmentation for localization.
- MVTec-compatible AU-PRO, image AUROC, and ECE reporting.

The category-specific normal-reference memory used in earlier experiments is **not part of Phase 1**. Its implementation is retained under `backend/experimental/` only as a decision trace; its large tensor cache was removed.

### Working principle

> The system asks CLIP whether an image looks semantically defective, both globally and in local patches, and calibrates that score into a probability.

## Detection output

The typed `DetectionResult` contains:

- `defect_logit`: raw zero-shot language anomaly score.
- `defect_probability`: primary temperature-calibrated probability.
- `mc_mean_probability`: mean probability across MC Dropout passes.
- `mc_variance`: variance across MC Dropout passes.
- `mc_predictive_entropy`: predictive Bernoulli entropy, bounded by ln(2).
- `heatmap`: 7x7 anomaly map before visualization resizing.

MC Dropout is additional uncertainty metadata; it does **not** replace the calibrated primary probability.

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

Evaluation reports image AUROC, macro image AUROC, AU-PRO at FPR=0.30, ECE, and mean MC Dropout variance/entropy.

## Localization metric

Phase 1 uses the released MVTec-style PRO protocol: connected ground-truth regions are weighted equally, normal pixels determine FPR, and the PRO curve is integrated only up to FPR=0.30. The implementation uses exact score thresholds rather than the previous 200-threshold approximation.

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
    detection.py          # typed calibrated + MC uncertainty result
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
