# Phase 5 Results: Two-Stage Pipeline, Branch Comparison & Hyperparameter Freeze

- **Commit**: `f16ef0f`
- **Config Hash**: `a1d1c42e0c28`
- **Frozen Settings SHA-256**: `153e12546c82333b863b255349e6d2644805b97fb1444a4676f47ea5da9dc70b`
- **Calibration Status**: `calibrated, not strictly zero-shot` (`T=1.10119`, `bias=-1.14598`)
- **Selected Fusion Weights (5 Validation Categories)**: `w_d=0.70`, `w_c=0.30`
- **1920x1080 Memory & Determinism (`step5_memtest.py`)**: Peak VRAM = `628.4 MB` (< `7168 MB`), Deterministic = `True`

## 1. Macro Branch Comparison on 10 Held-Out Categories

| Branch / Mode | Macro Image AUROC | Macro Pixel AUROC | Macro AU-PRO@0.05 |
| :--- | :---: | :---: | :---: |
| `CLIP-only (ViT-L/14@336)` | `0.8629` | — | — |
| `DINO-only (knn_global, k=4)` | `0.9598` | — | — |
| `Stage A-only (Global Presence ROI)` | `0.5349` | — | — |
| **`Fused (Stage A + DINO + CLIP)`** | **`0.9658`** | **`0.9656`** | **`0.7362`** |

## 2. Per-Category Results on 10 Held-Out Categories (`k=4`)

| Category | N (Norm/Anom) | CLIP Img AUROC | DINO Img AUROC | Stage A Img AUROC | Fused Img AUROC | Fused Px AUROC | Fused PRO@0.05 | Recall@5%FRR | ECE | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `bottle` | 83 (20/63) | 0.9079 | 0.9952 | 0.4639 | **0.9984** | **0.9914** | 0.8067 | 1.0000 | 0.1047 |  |
| `cable` | 150 (58/92) | 0.8160 | 0.9183 | 0.6668 | **0.9496** | **0.9194** | 0.6427 | 0.7935 | 0.2129 |  |
| `capsule` | 132 (23/109) | 0.7467 | 0.9589 | 0.5523 | **0.9625** | **0.9797** | 0.7850 | 0.9083 | 0.2070 |  |
| `grid` | 78 (21/57) | 0.9557 | 1.0000 | 0.3375 | **1.0000** | **0.9943** | 0.8423 | 1.0000 | 0.1648 |  |
| `leather` | 124 (32/92) | 0.9990 | 1.0000 | 0.4883 | **1.0000** | **0.9844** | 0.8488 | 1.0000 | 0.0774 |  |
| `pill` | 167 (26/141) | 0.7390 | 0.9482 | 0.6435 | **0.9438** | **0.9040** | 0.8082 | 0.8014 | 0.2190 |  |
| `screw` | 160 (41/119) | 0.8918 | 0.7883 | 0.4048 | **0.8116** | **0.9765** | 0.6566 | 0.3109 | 0.4248 |  |
| `tile` | 117 (33/84) | 0.9939 | 0.9996 | 0.5325 | **0.9996** | **0.9606** | 0.5262 | 1.0000 | 0.1516 |  |
| `toothbrush` | 42 (12/30) | 0.7917 | 0.9917 | 0.7111 | **0.9944** | **0.9882** | 0.8023 | 0.9667 | 0.1344 | `[LOW-N]` |
| `zipper` | 151 (32/119) | 0.7878 | 0.9976 | 0.5479 | **0.9976** | **0.9573** | 0.6430 | 0.9832 | 0.0639 |  |
