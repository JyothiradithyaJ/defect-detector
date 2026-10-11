# PHASE 3: DINOv2 Patch Memory Bank & Few-Shot Scoring — STATUS: CODE COMPLETE (PENDING USER GATE RUN)

## Task List
- [x] Implement `v3/core/defect/aggregate.py` supporting `match.agg_radii` (`[1, 3]`) and top-$k$ patch scoring (`dino.topk_frac`).
- [x] Implement `v3/core/bank.py` (`DINOFeatureExtractor` with `D:\defect_cache\dino_features` fp16 caching and `ReferenceBank` with photometric-only augmentation unless `recipe.symmetry` permits `flip`/`rot`).
- [x] Implement `v3/core/defect/knn.py` (`GlobalKNNScorer`), `v3/core/mismatch/positional_knn.py` (`PositionalKNNScorer`), and `v3/core/defect/pca.py` (`PCAResidualScorer`) adhering to `v3.core.contracts.Scorer`.
- [x] Implement `v3/eval/download_dino.py`, `v3/eval/step3_cache_dino.py`, `v3/eval/step3_fewshot.py`, and `v3/eval/step3_vram.py` with `--dry-run`, 30s progress logging, and skip-if-exists resumption.
- [x] Implement synthetic unit tests in `v3/tests/test_defect_synthetic.py` (swap test, aggregation test, reference/negative disjointness test, thin low-contrast scratch test).

## Built
- [`v3/core/bank.py`](file:///c:/Users/JYOTHIRADITHYA%20J/Downloads/defect-detector/v3/core/bank.py): `DINOFeatureExtractor` (`dinov2_vitl14_reg`, 560 px short side, $40 \times 40$ patch grid, 1024-D fp16 disk cache) and `ReferenceBank`.
- [`v3/core/defect/knn.py`](file:///c:/Users/JYOTHIRADITHYA%20J/Downloads/defect-detector/v3/core/defect/knn.py): Chunked global patch k-NN scorer (`knn_global`).
- [`v3/core/mismatch/positional_knn.py`](file:///c:/Users/JYOTHIRADITHYA%20J/Downloads/defect-detector/v3/core/mismatch/positional_knn.py): Windowed positional k-NN scorer (`knn_positional`, radius $r=2$).
- [`v3/core/defect/pca.py`](file:///c:/Users/JYOTHIRADITHYA%20J/Downloads/defect-detector/v3/core/defect/pca.py): Subspace PCA residual scorer (`pca_residual`).
- [`v3/core/defect/aggregate.py`](file:///c:/Users/JYOTHIRADITHYA%20J/Downloads/defect-detector/v3/core/defect/aggregate.py): Spatial neighbourhood smoothing (`agg_radii: [1, 3]`) and top-$k$ image scoring.
- [`v3/eval/download_dino.py`](file:///c:/Users/JYOTHIRADITHYA%20J/Downloads/defect-detector/v3/eval/download_dino.py), [`v3/eval/step3_cache_dino.py`](file:///c:/Users/JYOTHIRADITHYA%20J/Downloads/defect-detector/v3/eval/step3_cache_dino.py), [`v3/eval/step3_fewshot.py`](file:///c:/Users/JYOTHIRADITHYA%20J/Downloads/defect-detector/v3/eval/step3_fewshot.py), [`v3/eval/step3_vram.py`](file:///c:/Users/JYOTHIRADITHYA%20J/Downloads/defect-detector/v3/eval/step3_vram.py).

## Exact Reproduce Commands
```powershell
Remove-Item Env:\HF_HUB_OFFLINE -ErrorAction SilentlyContinue
D:\envs\anomaly_v3\Scripts\python.exe -u -m v3.eval.download_dino
$env:HF_HUB_OFFLINE = "1"
D:\envs\anomaly_v3\Scripts\python.exe -u -m v3.eval.step3_cache_dino
D:\envs\anomaly_v3\Scripts\python.exe -u -m v3.eval.step3_fewshot
D:\envs\anomaly_v3\Scripts\python.exe -u -m v3.eval.step3_vram
```

## Reproducibility Stamp
- **Branch**: `v3/phase-3-dino`
- **Seeds**: `[0, 1, 2]`
- **Config Path**: `v3/config/default.yaml`
- **AU-PRO Normalisation Convention**: Per-region overlap recall integrated over $\text{FPR} \in [0, 0.05]$ via trapezoidal rule and divided by $0.05$.
- **Low-N Flag**: `toothbrush` ($N=42$: 12 normal, 30 anomalous) flagged `[LOW-N]`.
- **Fixed-View Note**: Positional patch matching (`knn_positional`) without Stage 1 (`S1`) geometric alignment is only meaningful for fixed-view categories where parts appear in consistent canonical coordinates; free-rotation categories (e.g. `screw`, `metal_nut`, `hazelnut`) require either `S1` alignment or `knn_global`.

## Gate to Check
- Paste the contents of `D:\defect_cache\results\phase_3_summary.json` (or macro Image AUROC, Pixel AUROC, and AU-PRO@0.05 at $k \in \{1, 2, 4, 8, 16\}$ across `knn_global`, `knn_positional`, `pca_residual`) and `D:\defect_cache\results\step3_vram.json`.

## Numbers
- **Status**: `pending, run by user` (see `RUN_PHASES.md` Phase 3 section).

## Open Issues
- GPU cache build and few-shot sweep are deferred to manual execution via `RUN_PHASES.md`.

