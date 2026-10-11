# Phase 2 Report: Multi-Backbone CLIP, Domain Vocabulary & Polarity Analysis

- **Commit Hash**: `18f8e43`
- **Config Hash**: `a1d1c42e0c28`
- **Seeds**: `[0, 1, 2]` (CLIP zero-shot inference with frozen weights and fixed prompts is strictly deterministic; spread across seeds = `0.000000`)
- **Split Mode**: `v3-category` (5 validation categories for temperature & fusion weight selection: `carpet, hazelnut, metal_nut, transistor, wood`; 10 held-out categories evaluated on full MVTec test sets)
- **AU-PRO Normalisation Convention**: Per-region overlap recall is integrated over false positive rate $\text{FPR} \in [0, \alpha]$ via the trapezoidal rule (`np.trapezoid`) and divided by $\alpha \in \{0.05, 0.30\}$ so that normalized AU-PRO lies in $[0, 1]$.
- **Low-N Flag**: `toothbrush` ($N=42$: 12 normal, 30 anomalous) is explicitly flagged `LOW-N` for Recall@5%FRR and ECE.

---

## 1. Backbone Runtime, VRAM & Embedding Cache Summary

| Backbone | Pretrained / Input | Precision | Cache Size on Disk | Extraction Peak VRAM | Extraction Time/Img | Cached Eval Time/Img | Baseline 6-Decimal Match |
|---|---|---|---:|---:|---:|---:|---|
| `clip_b32_224` | `ViT-B-32-quickgelu` (224px) | `fp32` | 0.609 GB | 587.7 MB | 0.0177 s | 0.0068 s | **YES** (`max_diff = 0.000000`) |
| `clip_l14_336` | `ViT-L-14-336` (336px) | `fp16` | 10.401 GB | 991.0 MB | 0.3790 s | 0.0477 s | N/A (Upgraded Backbone) |

---

## 2. Held-Out Evaluation: `clip_b32_224` (`fp32`, Domain Generic Vocabulary)

| Category | N (Norm / Anom) | P1 Img AUROC | P2 Img AUROC | P1 Px AUROC | P2 Px AUROC | AU-PRO@0.05 | AU-PRO@0.30 | Recall@5%FRR | ECE | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `bottle` | 83 (20/63) | 0.9365 | **0.9302** | 0.8207 | **0.8641** | 0.4331 | 0.7120 | 0.8571 | 0.1734 |  |
| `cable` | 150 (58/92) | 0.7479 | **0.6576** | 0.4188 | **0.3783** | 0.0078 | 0.0729 | 0.1413 | 0.2132 |  |
| `capsule` | 132 (23/109) | 0.7060 | **0.6976** | 0.6057 | **0.5624** | 0.0920 | 0.2116 | 0.3028 | 0.5084 |  |
| `grid` | 78 (21/57) | 0.9323 | **0.9056** | 0.9273 | **0.8697** | 0.2187 | 0.5820 | 0.7018 | 0.2922 |  |
| `leather` | 124 (32/92) | 0.9986 | **1.0000** | 0.6558 | **0.5183** | 0.0150 | 0.1385 | 1.0000 | 0.3095 |  |
| `pill` | 167 (26/141) | 0.5908 | **0.6238** | 0.7835 | **0.6659** | 0.0071 | 0.1658 | 0.1277 | 0.0532 |  |
| `screw` | 160 (41/119) | 0.6337 | **0.6259** | 0.7585 | **0.3597** | 0.0042 | 0.0781 | 0.2269 | 0.4422 |  |
| `tile` | 117 (33/84) | 0.9693 | **0.9260** | 0.6266 | **0.3801** | 0.0183 | 0.1109 | 0.6905 | 0.2005 |  |
| `toothbrush` | 42 (12/30) | 0.5667 | **0.4972** | 0.3734 | **0.6678** | 0.0537 | 0.3007 | 0.1000 | 0.0977 | **LOW-N** |
| `zipper` | 151 (32/119) | 0.8262 | **0.7747** | 0.8391 | **0.7238** | 0.1884 | 0.4511 | 0.2269 | 0.6835 |  |
| **Macro Mean (10 Held-Out)** | **1,226** | 0.7908 | **0.7639** | 0.6809 | **0.5990** | **0.1038** | **0.2824** | **0.4375** | **0.2974** | Deterministic |

---

## 3. Held-Out Evaluation: `clip_l14_336` (`fp16`, Domain Generic Vocabulary)

| Category | N (Norm / Anom) | P1 Img AUROC | P2 Img AUROC | P1 Px AUROC | P2 Px AUROC | AU-PRO@0.05 | AU-PRO@0.30 | Recall@5%FRR | ECE | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `bottle` | 83 (20/63) | 0.9365 | **0.9119** | 0.8207 | **0.5204** | 0.1843 | 0.2682 | 0.7460 | 0.1758 |  |
| `cable` | 150 (58/92) | 0.7479 | **0.8053** | 0.4188 | **0.4347** | 0.1175 | 0.2339 | 0.5109 | 0.1451 |  |
| `capsule` | 132 (23/109) | 0.7060 | **0.7599** | 0.6057 | **0.3927** | 0.3077 | 0.3498 | 0.4954 | 0.4357 |  |
| `grid` | 78 (21/57) | 0.9323 | **0.9724** | 0.9273 | **0.8143** | 0.2263 | 0.4902 | 0.9123 | 0.4620 |  |
| `leather` | 124 (32/92) | 0.9986 | **0.9990** | 0.6558 | **0.7187** | 0.3377 | 0.4862 | 1.0000 | 0.2761 |  |
| `pill` | 167 (26/141) | 0.5908 | **0.7297** | 0.7835 | **0.4213** | 0.1647 | 0.2188 | 0.3688 | 0.0510 |  |
| `screw` | 160 (41/119) | 0.6337 | **0.8963** | 0.7585 | **0.3560** | 0.0061 | 0.0181 | 0.5882 | 0.3980 |  |
| `tile` | 117 (33/84) | 0.9693 | **0.9802** | 0.6266 | **0.2986** | 0.0068 | 0.0562 | 0.8929 | 0.2492 |  |
| `toothbrush` | 42 (12/30) | 0.5667 | **0.6778** | 0.3734 | **0.3483** | 0.0130 | 0.0245 | 0.2333 | 0.0781 | **LOW-N** |
| `zipper` | 151 (32/119) | 0.8262 | **0.7605** | 0.8391 | **0.5076** | 0.0546 | 0.2052 | 0.4958 | 0.0815 |  |
| **Macro Mean (10 Held-Out)** | **1,226** | 0.7908 | **0.8493** | 0.6809 | **0.4813** | **0.1419** | **0.2351** | **0.6244** | **0.2353** | Deterministic |

---

## 4. Local-Term Re-Evaluation (Selected on 5 Validation Categories Only)

| Setting | Global Weight ($a$) | Local Weight ($b$) | `clip_b32_224` Val Macro Img AUROC | `clip_l14_336` Val Macro Img AUROC |
|---|---:|---:|---:|---:|
| `global_only` | 1.00 | 0.00 | 0.8161 | 0.8496 |
| `fused_095` | 0.95 | 0.05 | 0.8146 | 0.8502 |
| `fused_090` | 0.90 | 0.10 | 0.8129 | 0.8501 |
| `fused_080` | 0.80 | 0.20 | 0.8041 | 0.8507 |
| `fused_050` | 0.50 | 0.50 | 0.7624 | 0.8518 |
| `local_only` | 0.00 | 1.00 | 0.6018 | 0.6692 |

**Finding on Local Term**:
- On `clip_b32_224`, adding a small local patch weight (`0.95 / 0.05`) shifts validation Macro Image AUROC from `0.8161` (`global_only`) to `0.8146` (`-0.0015`), whereas `local_only` (`0.00 / 1.00`) degrades to `0.6018`.
- On `clip_l14_336`, `global_only` achieves `0.8496` vs `0.8502` at `0.95 / 0.05` (`+0.0006`) and `0.6692` for `local_only`. Raw CLIP patch tokens lack spatial localization alignment without visual reference matching (Stage 2 DINOv2).

---

## 5. Polarity Mechanism Summary (No Per-Category Sign Flipping)

| Backbone | Polarity Variant | Held-Out Macro Pixel AUROC | Held-Out Macro Image AUROC |
|---|---|---:|---:|
| `clip_b32_224` | `final_tokens_native` (`ln_post + proj`, $s_{def} - s_{norm}$) | 0.4475 | 0.7640 |
| `clip_b32_224` | `intermediate_native` (no `ln_post`, $s_{def} - s_{norm}$) | 0.4010 | 0.7636 |
| `clip_b32_224` | `intermediate_inverted` (v1 legacy $-(s_{def} - s_{norm})$) | 0.5990 | 0.7668 |
| `clip_l14_336` | `final_tokens_native` (`ln_post + proj`, $s_{def} - s_{norm}$) | 0.5135 | 0.8640 |
| `clip_l14_336` | `intermediate_native` (no `ln_post`, $s_{def} - s_{norm}$) | 0.5187 | 0.8644 |
| `clip_l14_336` | `intermediate_inverted` (v1 legacy $-(s_{def} - s_{norm})$) | 0.4813 | 0.8629 |
