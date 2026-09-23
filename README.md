# Defect Detector

ML-based industrial defect detection using a frozen CLIP ViT-B/32 backbone.

## Phase 1 — ML Core

- Frozen CLIP image and text encoder
- WinCLIP-style normal/anomalous prompt bank
- Global-local similarity fusion
- MC Dropout uncertainty estimation
- Temperature-scaling calibration
- MVTec AD evaluation using AUROC, AU-PRO, and ECE

## Requirements

- Python version: to be recorded after environment setup
- CPU-compatible PyTorch
- MVTec AD dataset stored locally in `data/mvtec_ad/`

## Repository layout

```text
backend/    ML core, scripts, tests, and future API
data/       Local dataset, split manifests, and generated prompt cache
artifacts/  Calibration files and evaluation outputs
```
