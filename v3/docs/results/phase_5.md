# PHASE 5: Fusion, Calibration, Decision & Report — STATUS: CODE COMPLETE (PENDING USER GATE RUN)

## Task List
- [x] Implement `v3/core/fusion.py` (`RobustNormalizer` and `BranchFuser` weighted sum).
- [x] Implement `v3/core/calibrate.py` (`LogisticCalibrator` fitted on labelled validation data; outputs called probabilities only after calibration; explicitly documented as "calibrated, not strictly zero-shot").
- [x] Implement `v3/core/decide.py` implementing README section S4 (`RECAPTURE`, `REJECT`, `REWORK`, `PASS`) and disjoint normal quantile threshold $\tau$.
- [x] Implement `v3/core/report.py` with `Result` contract validation and automatic commit/seed/config hash stamping on JSON and PNG outputs.
- [x] Implement `v3/core/pipeline.py` (`TwoStagePipeline` wiring `S0` to `S5`).
- [x] Implement `v3/eval/step5_compare.py`, `v3/eval/step5_freeze.py`, and `v3/eval/step5_memtest.py`.
- [x] Implement unit tests in `v3/tests/test_phase5_pipeline.py` (S4 truth table, `Result` contract validation, determinism check).

## Built
- [`v3/core/fusion.py`](file:///c:/Users/JYOTHIRADITHYA%20J/Downloads/defect-detector/v3/core/fusion.py), [`v3/core/calibrate.py`](file:///c:/Users/JYOTHIRADITHYA%20J/Downloads/defect-detector/v3/core/calibrate.py), [`v3/core/decide.py`](file:///c:/Users/JYOTHIRADITHYA%20J/Downloads/defect-detector/v3/core/decide.py), [`v3/core/report.py`](file:///c:/Users/JYOTHIRADITHYA%20J/Downloads/defect-detector/v3/core/report.py), [`v3/core/pipeline.py`](file:///c:/Users/JYOTHIRADITHYA%20J/Downloads/defect-detector/v3/core/pipeline.py).
- [`v3/eval/step5_compare.py`](file:///c:/Users/JYOTHIRADITHYA%20J/Downloads/defect-detector/v3/eval/step5_compare.py), [`v3/eval/step5_freeze.py`](file:///c:/Users/JYOTHIRADITHYA%20J/Downloads/defect-detector/v3/eval/step5_freeze.py), [`v3/eval/step5_memtest.py`](file:///c:/Users/JYOTHIRADITHYA%20J/Downloads/defect-detector/v3/eval/step5_memtest.py).

## Exact Reproduce Commands
```powershell
D:\envs\anomaly_v3\Scripts\python.exe -u -m v3.eval.step5_compare
D:\envs\anomaly_v3\Scripts\python.exe -u -m v3.eval.step5_freeze
D:\envs\anomaly_v3\Scripts\python.exe -u -m v3.eval.step5_memtest
```

## Reproducibility & Calibration Note
- **Branch**: `v3/phase-5-fusion`
- **Seeds**: `[0, 1, 2]`
- **Calibration Convention**: Score-to-probability conversion uses logistic temperature/bias fitted on the 5 labelled validation categories (`calibrated, not strictly zero-shot`). Uncalibrated outputs are always referred to as anomaly scores, never probabilities.
- **AU-PRO Normalisation Convention**: Integrated over $\text{FPR} \in [0, 0.05]$ and divided by $0.05$.
- **Low-N Flag**: `toothbrush` ($N=42$) is marked `[LOW-N]`.

## Gate to Check
- Paste the contents of:
  1. `D:\defect_cache\results\phase_5_summary.json` (CLIP-only vs DINO-only vs Stage A only vs Fused A+B macro & per-category metrics)
  2. `v3/config/frozen_settings.json` and `v3/config/frozen_settings.sha256`
  3. `D:\defect_cache\results\step5_memtest.json` (confirming $1920 \times 1080$ Peak VRAM $< 7.0\text{ GB}$ and determinism diff $= 0.0$).

## Numbers
- **Status**: `pending, run by user` (see `RUN_PHASES.md` Phase 5 section).

## Open Issues
- Freeze file `v3/config/frozen_settings.json` must be generated via `step5_freeze` before running Phase 6 (`step6_apd_protocol`).

