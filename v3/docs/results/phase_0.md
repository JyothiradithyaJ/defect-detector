# PHASE 0: Setup and freeze  STATUS: DONE

## Built
- Established v3 package skeleton per `README_v3.md` section 9: `config/`, `core/`, `eval/`, `tests/`, and `docs/`.
- Defined pure data contracts (`ROI`, `Recipe`, `Mismatch`, `Result`) and `Scorer` protocol in `v3/core/contracts.py`.
- Configured `.gitignore` for v3 logs (`v3/docs/results/*_logs/`) and weights, and pinned `v3/requirements.txt`.
- Initialized local Python 3.12 virtual environment via `uv` with PyTorch, OpenCLIP, scikit-learn, OpenCV, pytest, and ruff.
- Created and pushed tag `v1-baseline` at base commit `232f9d3` on main.
- Created branch `v3/phase-0-setup` branching from `v3/two-stage` (scaffold commit `45ba16b`).

## Gate
- `pytest` on `v3/tests/`: **PASS** (3 passed, 4 skipped stubs in 0.29s).
- `ruff check v3/`: **PASS** (0 errors).
- v1 baseline command reproducibility verified via smoke evaluation on `bottle` category: **PASS** (Overall image AUROC: 0.9338).
- MVTec AD dataset directory and category integrity check: **PASS** (15 categories verified with complete train/test sets).

## Numbers
- Baseline smoke evaluation (`bottle` category):
  - Image AUROC: 0.9338
  - Pixel AUROC: 0.8344
  - Pixel AU-PRO: 0.6957
  - ECE: 0.2300

## Reproducibility Record
- **Commit hash**: `45ba16b` (base commit `232f9d3`)
- **Branch**: `v3/phase-0-setup` (working branch off `v3/two-stage`)
- **Tag**: `v1-baseline` (`232f9d3`)
- **Seed**: `[0, 1, 2]`
- **Config hash**: `a4677e3de4b8bb35005837d8dfc016ab245ac10f6c011d15a5823763606e4e73` (`v3/config/default.yaml`)
- **OS**: Windows 11 (AMD64)
- **Python**: 3.12.14
- **Torch version**: 2.5.1+cpu
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB VRAM), Driver 566.07, CUDA 12.7 (smoke test executed on CPU per prompt protocol)
- **Date**: 2026-10-09

## Decisions Logged (5)
1. Keep all implementation under `v3/` without editing legacy `v1` files.
2. Tag `v1-baseline` at `232f9d3`; develop on `v3/two-stage` via short-lived phase branches (`v3/phase-N-<name>`).
3. Manage `.venv` with Python 3.12 via `uv` and pin versions in `v3/requirements.txt`.
4. Document and verify exact v1 reproduction commands.
5. Encode pipeline contracts strictly in `v3/core/contracts.py`.

## Risks / Open Issues
- Full 15-category v1 baseline execution in Phase 1 requires either ~8-9 minutes on CPU or running on CUDA GPU with explicit confirmation.
- APD dataset access and license must be confirmed before Phase 6.

## Next
Phase 1: Evaluation harness (`eval/harness.py`, `metrics.py`, `splits.py`) and full v1 baseline comparison table, awaiting go-ahead.
