# v3 Decision Log

## 2026-10-09 — Phase 0 setup and freeze
- **Decision:** Establish `v3/` as an isolated pipeline implementation with zero modifications to legacy `v1` files.
  **Reason:** Required by `README_v3.md` and `BUILD_PROMPT.md` to preserve the v1 baseline without regression.
- **Decision:** Tag base commit `232f9d3` as `v1-baseline` and base the `v3/two-stage` branch from it. Phase work happens on `v3/phase-N-<name>` branches.
  **Reason:** Git branch management protocol in addendum item 1.
- **Decision:** Use Python 3.12 via `uv` in `.venv` with pinned packages in `v3/requirements.txt`.
  **Reason:** System default Python 3.14 lacks stable binary wheels for PyTorch and C-extensions on Windows; Python 3.12 provides full compatibility.
- **Decision:** Record exact v1 reproduction commands:
  ```bash
  .venv/Scripts/python backend/scripts/create_mvtec_manifests.py
  .venv/Scripts/python backend/scripts/fit_temperature.py --device cpu
  .venv/Scripts/python backend/scripts/evaluate_mvtec.py --device cpu
  ```
  **Reason:** Establishes reproducible baseline verification command sequence for Phase 1.
- **Decision:** Define core contracts (`ROI`, `Recipe`, `Mismatch`, `Result`, `Scorer`) in `v3/core/contracts.py` as pure immutable dataclasses and protocols.
  **Reason:** Enforces contract boundaries across stages S0 to S5 per `README_v3.md` section 5.
- **Open:** Confirm APD dataset availability and license before Phase 6. No downloading or URL guessing allowed.

## 2026-10-09 — Phase 1 evaluation harness and split definitions
- **Decision:** v3 category split rule for MVTec AD validation vs evaluation categories:
  - **Rule**: Stratified random sampling using seed 42. Categories are sorted alphabetically, divided into 5 textures (`carpet`, `grid`, `leather`, `tile`, `wood`) and 10 objects (`bottle`, `cable`, `capsule`, `hazelnut`, `metal_nut`, `pill`, `screw`, `toothbrush`, `transistor`, `zipper`). Sample 2 textures and 3 objects (`random.Random(42).sample`) for validation. The remaining 10 categories are held out for official evaluation.
  - **Validation categories (5)**: `['carpet', 'hazelnut', 'metal_nut', 'transistor', 'wood']`
  - **Evaluation categories (10)**: `['bottle', 'cable', 'capsule', 'grid', 'leather', 'pill', 'screw', 'tile', 'toothbrush', 'zipper']`
  - **Reason**: Guarantees zero leakage during temperature scaling and hyperparameter tuning; calibration is strictly fitted on validation categories only.
- **Decision:** Support two evaluation modes in harness:
  1. `v1-native`: Uses v1's exact stratified manifests (`calibration.json` and `evaluation.json`) to verify 1:1 bit-level parity against the published v1 baseline.
  2. `v3-category`: Calibrates on the 5 validation categories and evaluates on the 10 held-out categories as the official v3 baseline protocol.

## 2026-10-09 — Phase 2 CLIP upgrade and defect vocabulary
- **Decision:** Environment upgrade to CUDA cu124:
  - `torch==2.6.0+cu124`
  - `torchvision==0.21.0+cu124`
  - `open_clip_torch==3.3.0`
  - CUDA Driver `566.07`, CUDA Version `12.7`, GPU: `NVIDIA GeForce RTX 4060 Laptop GPU` (8 GB VRAM).
  - Model weights cache redirected to `D:\hf_cache` and `D:\torch_cache` where 216 GB free space is available.
- **Decision:** Frozen domain defect vocabulary defined in `v3/core/vocabulary.py`:
  - Formulated strictly from physical manufacturing domain knowledge of materials and assembly structure (e.g. cracked glass, cut insulation, broken bristle, bent pin).
  - Explicit rule: ZERO defect names read from the test split directories.
- **Decision:** Visual embedding disk cache strategy:
  - Key = `sha256(f"{backbone}:{input_res}:{precision}:{image_bytes_sha256}")`.
  - Cached files stored under `data/cache/visual_embeddings/` (gitignored).
  - Stores global visual embedding, final-layer spatial patch tokens, intermediate layer tokens, and horizontal flip embeddings.
- **Decision:** Polarity mechanism evaluation:
  - Strict rule: NO per-category sign flipping.
  - Evaluates final-layer projected patch tokens, intermediate-layer projected tokens without LayerNorm, and LayerNorm-normalized patch tokens.

## 2026-10-10 — Storage relocation to D: drive
- **Decision:** Relocate ML caches, virtual environment, datasets, and embeddings to drive D:
  - Base cache folder: `D:\ml_cache` with subfolders `hf`, `torch`, `uv`, `pip`, `tmp`.
  - Persisted user-level env vars via `setx`: `HF_HOME`, `TORCH_HOME`, `UV_CACHE_DIR`, `PIP_CACHE_DIR`, `TEMP`, `TMP`, `UV_LINK_MODE=copy`.
  - Model weights explicit cache: `cache_dir="D:/ml_cache/torch"` passed directly to `open_clip`.
  - Virtual environment relocated to: `D:\envs\anomaly_v3` with `torch==2.6.0+cu124` and `torchvision==0.21.0+cu124`.
  - Dataset relocated to `D:\data\mvtec_ad` (4.91 GB).
  - Visual embeddings cache configured to `D:\defect_cache\visual_embeddings` with uncompressed `np.savez` and auto-healing `np.load`.

- **Step A (CUDA & Env Check)**: status=PASS, elapsed=2.03s (GPU=NVIDIA GeForce RTX 4060 Laptop GPU, free_vram=6.93GB)
- **Step B (Baseline 6-Decimal Check)**: status=PASS, elapsed=0.01s (Matched to 6 decimals (511.5s run verified))

## 2026-10-11 — Phase 2 standalone scripts and offline checkpoint resolution
- **Last Completed Step**:
  - Step A (`CUDA & Env Check`) and Step B (`Baseline 6-Decimal Reproduction`) completed and verified (`max_diff = 0.000000`).
  - `clip_b32_224` visual embeddings (1,725/1,725 images) and `clip_l14_336` visual embeddings (~1,100/1,725 images) are cached in `D:\defect_cache\visual_embeddings`.
  - Background runner `task-1306` was stopped during Step C/D initialization.
- **Decision**: Resolve local OpenCLIP checkpoints directly from `D:\ml_cache\hf\hub\models--timm--*\snapshots\*\open_clip_model.safetensors` with `HF_HUB_OFFLINE=1` in `v3/core/features.py`.
  - **Reason**: Passing `cache_dir="D:/ml_cache/torch"` to `open_clip` caused `huggingface_hub` to look in `D:\ml_cache\torch` instead of `D:\ml_cache\hf\hub` (where the 1,632.5 MB `ViT-L-14-336` and 577.1 MB `ViT-B-32` `.safetensors` files already exist), triggering an unwanted re-download attempt. Direct snapshot path resolution guarantees zero network downloads.
- **Decision**: Provide standalone, config-driven scripts for Steps A–E (`v3/eval/step_a_env_check.py`, `v3/eval/step_b_baseline.py`, `v3/eval/step_c_smoke.py`, `v3/eval/step_d_cache_l14.py`, `v3/eval/step_e_evaluate.py`) and execution guide `RUN_COMMANDS.md` in the repo root.
- **Step A (CUDA & Env Check)**: status=PASS, elapsed=0.27s (GPU=NVIDIA GeForce RTX 4060 Laptop GPU, driver=566.07, free_vram=6.93GB, torch=2.6.0+cu124)
- **Step B (Baseline 6-Decimal Check)**: status=FAIL, elapsed=0.03s (max_diff=0.261665)
- **Step B (Baseline 6-Decimal Check)**: status=FAIL, elapsed=0.04s (Baseline check FAILED with max_diff=0.261665 > 1e-6)
- **Step C (Smoke Run 20 imgs/backbone)**: status=PASS, elapsed=18.37s (B32_VRAM=684.0MB, L14_VRAM=889.0MB)
- **Step D (L14 fp16 Embedding Cache Build)**: status=PASS, elapsed=121.94s (L14_size=10.401GB, L14_VRAM=991.0MB, sec_per_img=0.3790s)
- **Step E (Phase 2 Cached Evaluations & Reports)**: status=FAIL, elapsed=66.21s (CUDA error: an illegal memory access was encountered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
)
- **Step B (Baseline 6-Decimal Check)**: status=PASS, elapsed=0.05s (max_diff=0.000000)

## 2026-10-11 — Write-Only Mode Overrides for Phases 3–6
- **Decision (Session Override)**:
  1. `BUILD_PROMPT.md` Sections 5–6 "run pytest and the gate before DONE" is replaced by: write the code and synthetic unit tests, run only `ruff check v3/`, `python -m py_compile`, and `pytest --collect-only` (60 s cap); all GPU/model/evaluation gates are executed by the user via `RUN_PHASES.md`.
  2. `BUILD_PROMPT.md` Section 6 step 7 "stop after each phase" is replaced by: commit on stacked branches (`v3/phase-3-dino` -> `v3/phase-4-stage-a` -> `v3/phase-5-fusion` -> `v3/phase-6-apd`) and continue to the next phase, stopping only after Phase 6 or on a Section 8 condition.
  3. All 12 Hard Rules of `BUILD_PROMPT.md` remain strictly in force.
