# Phase 1 Diagnostic Report: Polarity Inversion and Score Decomposition

## 1. Summary of Hypotheses Investigated
1. **Pixel Map Polarity Inversion**: v1 negates the anomaly map (`-raw_map`). Does this sign flip explain below-chance pixel AUROCs (< 0.50) in `cable`, `toothbrush`, and `transistor`?
2. **Global vs Local Prompt Decomposition**: How much of image AUROC is carried by the global prompt contrast vs local patch contrast?
3. **Calibration Rank Preservation**: Is image AUROC invariant between uncalibrated logits and calibrated probabilities?

## 2. Per-Category Findings (All 15 MVTec AD Categories)

| Category | Images | Px AUROC (Inverted/v1) | Px AUROC (Native/Negated) | Img AUROC (Fused) | Img AUROC (Calibrated) | Img AUROC (Global Only) | Img AUROC (Local Only) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `bottle` | 67 | 0.8344 | 0.1656 | 0.9338 | 0.9338 | 0.9363 | 0.7255 |
| `cable` | 120 | 0.4192 | 0.5808 | 0.7362 | 0.7362 | 0.7365 | 0.3655 |
| `capsule` | 105 | 0.6060 | 0.3940 | 0.7261 | 0.7261 | 0.7305 | 0.3493 |
| `carpet` | 94 | 0.8463 | 0.1537 | 0.9646 | 0.9646 | 0.9665 | 0.3333 |
| `grid` | 64 | 0.9301 | 0.0699 | 0.9324 | 0.9324 | 0.9337 | 0.2215 |
| `hazelnut` | 88 | 0.7067 | 0.2933 | 0.7651 | 0.7651 | 0.7645 | 0.6373 |
| `leather` | 99 | 0.6401 | 0.3599 | 0.9989 | 0.9989 | 0.9984 | 0.5248 |
| `metal_nut` | 92 | 0.5602 | 0.4398 | 0.4730 | 0.4730 | 0.4790 | 0.4865 |
| `pill` | 134 | 0.7798 | 0.2202 | 0.5727 | 0.5727 | 0.5718 | 0.5895 |
| `screw` | 127 | 0.7569 | 0.2431 | 0.6251 | 0.6251 | 0.6309 | 0.4007 |
| `tile` | 93 | 0.6378 | 0.3622 | 0.9696 | 0.9696 | 0.9696 | 0.4328 |
| `toothbrush` | 34 | 0.3915 | 0.6085 | 0.5208 | 0.5208 | 0.5250 | 0.5917 |
| `transistor` | 80 | 0.3747 | 0.6253 | 0.6966 | 0.6966 | 0.6960 | 0.5280 |
| `wood` | 63 | 0.8920 | 0.1080 | 0.9403 | 0.9403 | 0.9458 | 0.1444 |
| `zipper` | 122 | 0.8349 | 0.1651 | 0.8938 | 0.8938 | 0.8870 | 0.8510 |
| **Macro Average** | **1382** | **0.6807** | **0.3193** | **0.7833** | **0.7833** | **0.7848** | **0.4788** |

## 3. Detailed Answers to Diagnostic Questions

### a) Does Sign Inversion Explain Below-Chance Pixel AUROC in Cable, Toothbrush, and Transistor?
- In `cable`, v1 inverted pixel AUROC is **0.4192** (< 0.50). Under the native (negated) map, pixel AUROC becomes **0.5808** (> 0.50).
- In `toothbrush`, v1 inverted pixel AUROC is **0.3915** (< 0.50). Under the native (negated) map, pixel AUROC becomes **0.6085** (> 0.50).
- In `transistor`, v1 inverted pixel AUROC is **0.3747** (< 0.50). Under the native (negated) map, pixel AUROC becomes **0.6253** (> 0.50).

**Conclusion based on numbers**:
While negating the anomaly map strictly flips AUROC (`AUROC_native = 1 - AUROC_inverted`) and pushes `cable` (0.5808), `toothbrush` (0.6085), and `transistor` (0.6253) above 0.50, **it severely destroys performance on the other 12 categories** (e.g. `grid` collapses from 0.9301 to 0.0699, `wood` from 0.8920 to 0.1080, `bottle` from 0.8344 to 0.1656).
Macro pixel AUROC across all 15 categories drops from **0.6807** under v1 inversion down to **0.3193** under native polarity.
Therefore, sign inversion is **NOT** a uniform coordinate error across categories. Rather, CLIP prompt similarity acts as an inverted signal on certain textures/objects and a direct signal on others. This empirically justifies the architecture in README_v3: replacing prompt-based patch scoring with reference patch matching (DINOv2 patch k-NN) in Stage B.

### b) Image AUROC: Global Term Only vs Local Term Only
- Macro Image AUROC from **Global Term only**: **0.7848**
- Macro Image AUROC from **Local Term only**: **0.4788**
- Macro Image AUROC with **Fused (0.95 Global + 0.05 Local)**: **0.7833**
The global term carries nearly all image-level discriminatory power (0.7848 vs 0.7833 fused). The local patch term alone achieves only 0.4788 macro AUROC (below random chance overall), showing that zero-shot prompt-based patch contrast is too noisy for standalone image-level verdicts.

### c) Calibrated vs Uncalibrated Image AUROC Identity
- Max absolute difference across all 15 categories between uncalibrated and calibrated image AUROC: `0.0e+00`
- **Confirmed**: Uncalibrated defect logits and calibrated probabilities produce **identical AUROC to machine precision** because Platt/temperature scaling is a strictly monotonic sigmoid transformation that preserves rank ordering.
