# PHASE 1: Evaluation harness and v1 baseline  STATUS: DONE

## Built
- Implemented `v3/eval/metrics.py` providing image AUROC, pixel AUROC, pixel AU-PRO (at 0.05 FPR and 0.30 FPR limits), recall at 5% FRR, and ECE.
- Implemented `v3/eval/splits.py` supporting `v1-native` and stratified `v3-category` splits with Rule 3 leakage assertions.
- Implemented `v3/eval/harness.py` with `V1BaselineScorer` adapting v1 CLIP zero-shot detector into the pipeline interface.
- Un-skipped and passed `v3/tests/test_leakage.py` validating reference/negative disjointness assertions.
- Executed full 15-category MVTec AD evaluation on CPU for both `v1-native` (reproducing main 1:1) and `v3-category` protocols.

## Gate
- `v1-native` evaluation reproduction: **PASS** (100% bit-level agreement across all 15 categories with `main:artifacts/evaluation-final/results.csv`).
- Leakage assertions on reference normals vs test negatives: **PASS**.
- Full test suite execution: **PASS** (5 passed, 3 skipped in 0.29s).
- Code lint check (`ruff check v3/`): **PASS** (0 errors).

## Numbers

### Mode 1: v1-native Split (15 Categories, Reproduction Verification)
*Evaluated on CPU using v1's native manifests (1382 test samples across 15 categories, calibrated on 343 calibration samples).*

| Category | Images | Normal | Anomaly | Image AUROC | Pixel AUROC | AU-PRO (0.05) | AU-PRO (0.30) | Recall@5%FRR | ECE | Diff vs Main (`results.csv`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `bottle` | 67 | 16 | 51 | 0.933824 | 0.834352 | 0.400356 | 0.695731 | 0.843137 | 0.229957 | +0.000000 |
| `cable` | 120 | 46 | 74 | 0.736193 | 0.419203 | 0.089428 | 0.190608 | 0.148649 | 0.179737 | +0.000000 |
| `capsule` | 105 | 18 | 87 | 0.726054 | 0.605988 | 0.105151 | 0.307831 | 0.298851 | 0.220559 | +0.000000 |
| `carpet` | 94 | 22 | 72 | 0.964646 | 0.846288 | 0.250753 | 0.559651 | 0.888889 | 0.280599 | +0.000000 |
| `grid` | 64 | 17 | 47 | 0.932416 | 0.930138 | 0.434348 | 0.798865 | 0.765957 | 0.231637 | +0.000000 |
| `hazelnut` | 88 | 32 | 56 | 0.765067 | 0.706712 | 0.140247 | 0.270659 | 0.446429 | 0.252244 | +0.000000 |
| `leather` | 99 | 26 | 73 | 0.998946 | 0.640106 | 0.168299 | 0.421483 | 0.986301 | 0.337712 | +0.000000 |
| `metal_nut`| 92 | 18 | 74 | 0.472973 | 0.560152 | 0.087088 | 0.303446 | 0.229730 | 0.018454 | +0.000000 |
| `pill` | 134 | 21 | 113 | 0.572693 | 0.779835 | 0.244529 | 0.563704 | 0.141593 | 0.049566 | +0.000000 |
| `screw` | 127 | 33 | 94 | 0.625081 | 0.756885 | 0.053462 | 0.350444 | 0.287234 | 0.028292 | +0.000000 |
| `tile` | 93 | 26 | 67 | 0.969575 | 0.637774 | 0.175940 | 0.357849 | 0.865672 | 0.282341 | +0.000000 |
| `toothbrush`| 34 | 10 | 24 | 0.520833 | 0.391489 | 0.037217 | 0.159267 | 0.166667 | 0.129431 | +0.000000 |
| `transistor`| 80 | 48 | 32 | 0.696615 | 0.374692 | 0.113988 | 0.289904 | 0.250000 | 0.308611 | +0.000000 |
| `wood` | 63 | 15 | 48 | 0.940278 | 0.892006 | 0.309173 | 0.752526 | 0.812500 | 0.225424 | +0.000000 |
| `zipper` | 122 | 26 | 96 | 0.893830 | 0.834854 | 0.360075 | 0.645689 | 0.552083 | 0.244612 | +0.000000 |
| **Macro** | **1382** | **374** | **1008** | **0.7833** | **0.6807** | **0.1980** | **0.4445** | **0.5122** | **0.2013** | **0.0000** |

*Determinism across seeds*: The v1 model on CPU is fully deterministic; seeds `[0, 1, 2]` yield mean values with spread = `0.0000`.

---

### Mode 2: Official v3-category Split (Held-Out Evaluation)
*Calibrated on 5 validation categories (`carpet`, `hazelnut`, `metal_nut`, `transistor`, `wood` — 521 samples). Fitted temperature: `0.011036`, bias: `0.418684`. Evaluated on 10 held-out categories (942 samples).*

| Category | Images | Normal | Anomaly | Image AUROC | Pixel AUROC | AU-PRO (0.05) | AU-PRO (0.30) | Recall@5%FRR | ECE | Sec/Image |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `bottle` | 67 | 16 | 51 | 0.933824 | 0.834352 | 0.400356 | 0.695731 | 0.843137 | 0.184917 | 0.2214 |
| `cable` | 120 | 46 | 74 | 0.736193 | 0.419203 | 0.089428 | 0.190608 | 0.148649 | 0.142976 | 0.1979 |
| `capsule` | 105 | 18 | 87 | 0.726054 | 0.605988 | 0.105151 | 0.307831 | 0.298851 | 0.392544 | 0.1971 |
| `grid` | 64 | 17 | 47 | 0.932416 | 0.930138 | 0.434348 | 0.798865 | 0.765957 | 0.343312 | 0.1832 |
| `leather` | 99 | 26 | 73 | 0.998946 | 0.640106 | 0.168299 | 0.421483 | 0.986301 | 0.357304 | 0.2027 |
| `pill` | 134 | 21 | 113 | 0.572693 | 0.779835 | 0.244529 | 0.563704 | 0.141593 | 0.100559 | 0.1800 |
| `screw` | 127 | 33 | 94 | 0.625081 | 0.756885 | 0.053462 | 0.350444 | 0.287234 | 0.101843 | 0.1818 |
| `tile` | 93 | 26 | 67 | 0.969575 | 0.637774 | 0.175940 | 0.357849 | 0.865672 | 0.279518 | 0.1938 |
| `toothbrush`| 34 | 10 | 24 | 0.520833 | 0.391489 | 0.037217 | 0.159267 | 0.166667 | 0.094902 | 0.1975 |
| `zipper` | 122 | 26 | 96 | 0.893830 | 0.834854 | 0.360075 | 0.645689 | 0.552083 | 0.346437 | 0.1838 |
| **Macro** | **942** | **239** | **703** | **0.7909** | **0.6831** | **0.2069** | **0.4491** | **0.5056** | **0.2344** | **0.1939** |

---

## Reproducibility Record
- **Branch**: `v3/phase-1-harness` (branched from `v3/two-stage` at commit `9de4e92`)
- **Seeds**: `[0, 1, 2]`
- **Config Hash**: `a4677e3de4b8bb35005837d8dfc016ab245ac10f6c011d15a5823763606e4e73` (`v3/config/default.yaml`)
- **Operating System**: Windows 11 (AMD64)
- **Python Version**: 3.12.14
- **PyTorch Version**: 2.5.1+cpu
- **Execution Device**: CPU
- **Date**: 2026-10-09

## Decisions Logged
1. Stratified random sampling rule (seed 42, 2 textures, 3 objects) for validation vs held-out evaluation categories.
2. Zero data leakage protocol: temperature fitting on validation categories only.
3. Dual-mode harness architecture (`v1-native` for regression verification and `v3-category` for official benchmarking).

## Risks / Open Issues
- v1 baseline Macro Image AUROC of 78.33% (v1-native) and 79.09% (v3-category) lags behind WinCLIP reference points (91.8% MVTec-AD). As diagnosed in README_v3 section 0 & 3, the inverted heatmap polarity and weak CLIP patch text prior cause significant degradation on tiny defects and textures (e.g. `metal_nut`: 47.3%, `toothbrush`: 52.1%, `pill`: 57.3%). Phase 2 and 3 (upgraded CLIP 336px + DINOv2 patch branch) are specifically designed to address these gaps.

## Next
Phase 1 complete. Awaiting user review and authorization to merge `v3/phase-1-harness` into `v3/two-stage`.
