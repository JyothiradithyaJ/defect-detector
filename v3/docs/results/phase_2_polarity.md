# Phase 2 Polarity Mechanism Diagnostic (10 Held-Out Categories)

Strict Rule: **Zero per-category sign flipping.** Compares final-layer projected patch tokens (`ln_post + proj`) against un-normalized intermediate patch projections and v1's globally inverted intermediate tokens.

## Per-Category Pixel AUROC by Polarity Variant

| Category | B32 `final_tokens_native` | B32 `intermediate_native` | B32 `intermediate_inverted` | L14 `final_tokens_native` | L14 `intermediate_native` | L14 `intermediate_inverted` |
|---|---:|---:|---:|---:|---:|---:|
| `bottle` | 0.1422 | 0.1359 | 0.8641 | 0.4768 | 0.4796 | 0.5204 |
| `cable` | 0.6373 | 0.6217 | 0.3783 | 0.5695 | 0.5653 | 0.4347 |
| `capsule` | 0.5314 | 0.4376 | 0.5624 | 0.6092 | 0.6073 | 0.3927 |
| `grid` | 0.1371 | 0.1303 | 0.8697 | 0.1866 | 0.1857 | 0.8143 |
| `leather` | 0.5238 | 0.4817 | 0.5183 | 0.2556 | 0.2813 | 0.7187 |
| `pill` | 0.3986 | 0.3341 | 0.6659 | 0.5792 | 0.5787 | 0.4213 |
| `screw` | 0.6969 | 0.6403 | 0.3597 | 0.6272 | 0.6440 | 0.3560 |
| `tile` | 0.6158 | 0.6199 | 0.3801 | 0.6974 | 0.7014 | 0.2986 |
| `toothbrush` | 0.4859 | 0.3322 | 0.6678 | 0.6450 | 0.6517 | 0.3483 |
| `zipper` | 0.3060 | 0.2762 | 0.7238 | 0.4883 | 0.4924 | 0.5076 |
| **Macro Mean** | **0.4475** | **0.4010** | **0.5990** | **0.5135** | **0.5187** | **0.4813** |
