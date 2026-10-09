# PHASE 1: Evaluation harness and v1 baseline  STATUS: DONE

## Built
- Implemented `v3/eval/metrics.py` providing image AUROC, pixel AUROC, pixel AU-PRO (at 0.05 FPR and 0.30 FPR limits), recall at 5% FRR, and ECE.
- Implemented `v3/eval/splits.py` supporting `v1-native` and stratified `v3-category` splits with Rule 3 leakage assertions.
- Implemented `v3/eval/harness.py` with `V1BaselineScorer` adapting v1 CLIP zero-shot detector into the pipeline interface.
- Un-skipped and passed `v3/tests/test_leakage.py` validating reference/negative disjointness assertions.
- Implemented `v3/eval/diagnostic.py` and generated `v3/docs/results/phase_1_diagnostic.md` assessing polarity negation, global vs local term contributions, and calibration rank invariance.
- Executed full MVTec AD evaluation on CPU for both `v1-native` (15 categories, 1382 test samples) and `v3-category` (10 held-out categories, 1204 full test set samples).

## Gate
- `v1-native` evaluation reproduction: **PASS** (matches to 6 decimals across all 15 categories with `main:artifacts/evaluation-final/results.csv`).
- Leakage assertions on reference normals vs test negatives: **PASS**.
- Full test suite execution: **PASS** (5 passed, 3 skipped in 0.29s).
- Code lint check (`ruff check v3/`): **PASS** (0 errors).

## Numbers

### Mode 1: v1-native Split (15 Categories, Regression Verification)
*Evaluated on CPU using v1's native manifests (1382 test samples across 15 categories, calibrated on 343 calibration samples).*

| Category | Images | Normal | Anomaly | Image AUROC | Pixel AUROC | AU-PRO (0.05) | AU-PRO (0.30) | Recall@5%FRR | ECE | Diff vs Main (`results.csv`) | Peak VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `bottle` | 67 | 16 | 51 | 0.933824 | 0.834352 | 0.400356 | 0.695731 | 0.843137 | 0.229957 | matches to 6 decimals | N/A (CPU run) |
| `cable` | 120 | 46 | 74 | 0.736193 | 0.419203 | 0.089428 | 0.190608 | 0.148649 | 0.179737 | matches to 6 decimals | N/A (CPU run) |
| `capsule` | 105 | 18 | 87 | 0.726054 | 0.605988 | 0.105151 | 0.307831 | 0.298851 | 0.220559 | matches to 6 decimals | N/A (CPU run) |
| `carpet` | 94 | 22 | 72 | 0.964646 | 0.846288 | 0.250753 | 0.559651 | 0.888889 | 0.280599 | matches to 6 decimals | N/A (CPU run) |
| `grid` | 64 | 17 | 47 | 0.932416 | 0.930138 | 0.434348 | 0.798865 | 0.765957 | 0.231637 | matches to 6 decimals | N/A (CPU run) |
| `hazelnut` | 88 | 32 | 56 | 0.765067 | 0.706712 | 0.140247 | 0.270659 | 0.446429 | 0.252244 | matches to 6 decimals | N/A (CPU run) |
| `leather` | 99 | 26 | 73 | 0.998946 | 0.640106 | 0.168299 | 0.421483 | 0.986301 | 0.337712 | matches to 6 decimals | N/A (CPU run) |
| `metal_nut`| 92 | 18 | 74 | 0.472973 | 0.560152 | 0.087088 | 0.303446 | 0.229730 | 0.018454 | matches to 6 decimals | N/A (CPU run) |
| `pill` | 134 | 21 | 113 | 0.572693 | 0.779835 | 0.244529 | 0.563704 | 0.141593 | 0.049566 | matches to 6 decimals | N/A (CPU run) |
| `screw` | 127 | 33 | 94 | 0.625081 | 0.756885 | 0.053462 | 0.350444 | 0.287234 | 0.028292 | matches to 6 decimals | N/A (CPU run) |
| `tile` | 93 | 26 | 67 | 0.969575 | 0.637774 | 0.175940 | 0.357849 | 0.865672 | 0.282341 | matches to 6 decimals | N/A (CPU run) |
| `toothbrush`| 34 | 10 | 24 | 0.520833 | 0.391489 | 0.037217 | 0.159267 | 0.166667 | 0.129431 | matches to 6 decimals | N/A (CPU run) |
| `transistor`| 80 | 48 | 32 | 0.696615 | 0.374692 | 0.113988 | 0.289904 | 0.250000 | 0.308611 | matches to 6 decimals | N/A (CPU run) |
| `wood` | 63 | 15 | 48 | 0.940278 | 0.892006 | 0.309173 | 0.752526 | 0.812500 | 0.225424 | matches to 6 decimals | N/A (CPU run) |
| `zipper` | 122 | 26 | 96 | 0.893830 | 0.834854 | 0.360075 | 0.645689 | 0.552083 | 0.244612 | matches to 6 decimals | N/A (CPU run) |
| **Macro** | **1382** | **374** | **1008** | **0.7833** | **0.6807** | **0.1980** | **0.4445** | **0.5122** | **0.2013** | **0.0000** | **N/A (CPU run)** |

*Determinism across seeds*: The v1 baseline on CPU is deterministic. Seeds [0, 1, 2] yield identical scores (spread = 0.0000).

---

### Mode 2: Official v3-category Split (Full MVTec Test Sets on Held-Out Categories)
*Calibrated on 5 validation categories (`carpet`, `hazelnut`, `metal_nut`, `transistor`, `wood` — 521 samples). Fitted temperature: `0.011036`, bias: `0.418684`. Evaluated on 10 held-out categories across their FULL test sets (1,204 samples total).*

| Category | Images | Normal | Anomaly | Image AUROC | Pixel AUROC | AU-PRO (0.05) | AU-PRO (0.30) | Recall@5%FRR | ECE | Sec/Image | Peak VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `bottle` | 83 | 20 | 63 | 0.936508 | 0.820713 | 0.413931 | 0.696405 | 0.825397 | 0.195723 | 0.2552 | N/A (CPU run) |
| `cable` | 150 | 58 | 92 | 0.747939 | 0.418799 | 0.103561 | 0.213076 | 0.293478 | 0.163346 | 0.3054 | N/A (CPU run) |
| `capsule` | 132 | 23 | 109 | 0.706023 | 0.605738 | 0.105998 | 0.290143 | 0.293578 | 0.389908 | 0.3015 | N/A (CPU run) |
| `grid` | 78 | 21 | 57 | 0.932331 | 0.927308 | 0.420495 | 0.783263 | 0.754386 | 0.358086 | 0.2607 | N/A (CPU run) |
| `leather` | 124 | 32 | 92 | 0.998641 | 0.655759 | 0.148452 | 0.389252 | 1.000000 | 0.349538 | 0.2726 | N/A (CPU run) |
| `pill` | 167 | 26 | 141 | 0.590835 | 0.783547 | 0.228371 | 0.541275 | 0.141844 | 0.103970 | 0.2456 | N/A (CPU run) |
| `screw` | 160 | 41 | 119 | 0.633736 | 0.758471 | 0.067177 | 0.355555 | 0.218487 | 0.089948 | 0.2907 | N/A (CPU run) |
| `tile` | 117 | 33 | 84 | 0.969336 | 0.626581 | 0.177729 | 0.365706 | 0.857143 | 0.278519 | 0.3334 | N/A (CPU run) |
| `toothbrush`| 42 | 12 | 30 | 0.566667 | 0.373431 | 0.043102 | 0.177470 | 0.166667 | 0.055046 | 0.3022 | N/A (CPU run) |
| `zipper` | 151 | 32 | 119 | 0.826155 | 0.839147 | 0.362968 | 0.652322 | 0.294118 | 0.334732 | 0.3014 | N/A (CPU run) |
| **Macro** | **1204** | **298** | **906** | **0.7908** | **0.6809** | **0.2072** | **0.4464** | **0.4845** | **0.2319** | **0.2869** | **N/A (CPU run)** |

---

## Reproducibility Record
- **Branch**: `v3/phase-1-harness`
- **Seeds**: `[0, 1, 2]` (deterministic: spread = 0.0000)
- **Config Hash**: `a4677e3de4b8bb35005837d8dfc016ab245ac10f6c011d15a5823763606e4e73` (`v3/config/default.yaml`)
- **Operating System**: Windows 11 (AMD64)
- **Python Version**: 3.12.14
- **PyTorch Version**: 2.5.1+cpu
- **Execution Device**: CPU (Peak VRAM: N/A)
- **Date**: 2026-10-09

## Decisions Logged
1. Stratified random sampling rule (seed 42, 2 textures, 3 objects) for validation vs held-out evaluation categories.
2. Official v3 baseline evaluates held-out categories on their full test sets (1,204 images across 10 categories), excluding validation categories.
3. Zero data leakage protocol: calibration fitted exclusively on validation categories.
4. Dual-mode harness architecture (`v1-native` for regression verification and `v3-category` for official benchmarking).

## Risks / Open Issues
- v1 baseline Macro Image AUROC of 78.33% (v1-native) and 79.08% (v3-category full test sets) lags behind WinCLIP reference points (91.8%). As diagnosed in `v3/docs/results/phase_1_diagnostic.md`:
  1. The global prompt contrast carries 0.7848 image AUROC, but the local patch term contributes only 0.4788 (below random chance).
  2. Inverted patch polarity artificially fixes cable, toothbrush, and transistor, but severely destroys the other 12 categories, showing that text prompts cannot reliably localize pixel-level defects across varied surface textures.
  3. These findings directly justify the Phase 2 upgrade (OpenCLIP 336 px + per-category defect vocabularies) and Phase 3 (DINOv2 patch k-NN memory bank).

## Next
Phase 1 complete. Awaiting authorization to merge `v3/phase-1-harness` into `v3/two-stage`.
