# RUN_PHASES.md — Manual Execution & Gate Guide (Phases 2–6)

This guide covers running the completed code for **Phase 2 Finish (Step E)** and **Phases 3, 4, 5, and 6** across stacked branches (`v3/phase-3-dino` $\rightarrow$ `v3/phase-4-stage-a` $\rightarrow$ `v3/phase-5-fusion` $\rightarrow$ `v3/phase-6-apd`).

---

## 0. Prerequisites

### A. Activate `D:\envs\anomaly_v3` & Set D-Drive Environment Variables
Run in PowerShell from the repository root (`c:\Users\JYOTHIRADITHYA J\Downloads\defect-detector`):

```powershell
& "D:\envs\anomaly_v3\Scripts\Activate.ps1"

$env:HF_HOME = "D:\ml_cache\hf"
$env:TORCH_HOME = "D:\ml_cache\torch"
$env:UV_CACHE_DIR = "D:\ml_cache\uv"
$env:PIP_CACHE_DIR = "D:\ml_cache\pip"
$env:TEMP = "D:\ml_cache\tmp"
$env:TMP = "D:\ml_cache\tmp"
$env:UV_LINK_MODE = "copy"
$env:HF_HUB_OFFLINE = "1"
```

### B. One-Time DINOv2 Weight Download (`dinov2_vitl14_reg`)
Before running Phase 3 offline, temporarily enable network access to cache `dinov2_vitl14_reg` (~1.2 GB) into `D:\ml_cache\torch`, then immediately re-enable offline mode:

```powershell
git checkout v3/phase-6-apd
Remove-Item Env:\HF_HUB_OFFLINE -ErrorAction SilentlyContinue
python -m v3.eval.download_dino --dry-run
python -m v3.eval.download_dino
$env:HF_HUB_OFFLINE = "1"
```

### C. Expected Dataset & Cache Paths on `D:`
Configured in `v3/config/default.yaml`:
- `D:\data\mvtec_ad` — MVTec AD dataset
- `D:\data\visa` — VisA dataset (optional in Phase 3; script logs `VisA not found at D:/data/visa, skipping` if absent)
- `D:\data\mvtec_loco` — MVTec LOCO AD dataset (required for Phase 4 Stage A evaluation)
- `D:\data\apd` — Assembled Panel Dataset for EV charging stations (required for Phase 6; **Reminder**: verify APD license terms before commercial/public redistribution; never commit APD images to git)
- `D:\defect_cache\visual_embeddings` — Phase 2 CLIP visual embeddings
- `D:\defect_cache\dino_features` — Phase 3 DINOv2 patch & CLS caches
- `D:\defect_cache\results` — All CSV/JSON/Markdown evaluation outputs
- `D:\defect_cache\logs` — All timestamped execution logs

---

## 1. Phase 2 Finish — Step E (Fast CPU Evaluation on Cached CLIP Embeddings)

Step E has been updated to encode the 22 text prompts on CPU when all visual embeddings are already cached on `D:\defect_cache\visual_embeddings`, avoiding any GPU model reload and using a fast vectorized validation sweep.

### Commands
```powershell
git checkout v3/two-stage
python -m v3.eval.step_e_evaluate --dry-run
python -m v3.eval.step_e_evaluate
```
- **Expected Runtime**: ~15–45 seconds on cached `.npz` embeddings.
- **Log File**: `D:\defect_cache\logs\step_e_evaluate.log`
- **What Success Looks Like**: Log ends with `Step E COMPLETE.` and writes `D:\defect_cache\results\phase_2_summary.json`, `phase_2_results.md`, `phase_2_clip_b32_224.csv`, and `phase_2_clip_l14_336.csv`.
- **GATE (Paste back to chat)**: Paste the console summary and contents of `D:\defect_cache\results\phase_2_results.md`.
- **On Failure**: Check `D:\defect_cache\logs\step_e_evaluate.log` for any missing category `.npz` file in `D:\defect_cache\visual_embeddings`.

---

## 2. Phase 3 — Stage B2 (DINOv2 Patch kNN + PCA Residual)

### Step 3.1: Run Unit Tests
```powershell
git checkout v3/phase-3-dino
pytest -p no:cacheprovider v3/tests/test_defect_synthetic.py -v
```

### Step 3.2: Build DINOv2 Feature Cache (`dinov2_vitl14_reg` @ 518 px, fp16)
Because feature extraction across 15 MVTec categories takes >60 seconds, launch it as a background job and tail the log:

```powershell
python -m v3.eval.step3_cache_dino --dry-run

$proc = Start-Process -FilePath "D:\envs\anomaly_v3\Scripts\python.exe" `
  -ArgumentList "-m v3.eval.step3_cache_dino --batch-size 4" `
  -WorkingDirectory (Get-Location) -NoNewWindow -PassThru
Get-Content "D:\defect_cache\logs\step3_cache_dino.log" -Wait -Tail 20
```
- **Expected Runtime**: ~8–15 minutes on RTX 4060 Laptop GPU (resumable per category; skips existing `.npz` files).
- **Log File**: `D:\defect_cache\logs\step3_cache_dino.log`
- **What Success Looks Like**: Progress lines every 30s; finishes with `DINO Cache Summary: Peak VRAM = ... MB, Avg ms/img = ..., Disk = ... GB`.

### Step 3.3: Run Few-Shot Protocol ($K \in \{1, 2, 4, 8, 16\} \times 3\text{ seeds}$) & Ablations
```powershell
python -m v3.eval.step3_fewshot --dry-run

$proc = Start-Process -FilePath "D:\envs\anomaly_v3\Scripts\python.exe" `
  -ArgumentList "-m v3.eval.step3_fewshot" `
  -WorkingDirectory (Get-Location) -NoNewWindow -PassThru
Get-Content "D:\defect_cache\logs\step3_fewshot.log" -Wait -Tail 20
```
- **Expected Runtime**: ~3–8 minutes on cached features.
- **Log File**: `D:\defect_cache\logs\step3_fewshot.log`
- **Outputs**:
  - `D:\defect_cache\results\phase_3_fewshot.csv`
  - `D:\defect_cache\results\phase_3_summary.json`

### Step 3.4: Measure Peak VRAM & Latency vs. $K$
```powershell
python -m v3.eval.step3_vram --dry-run
python -m v3.eval.step3_vram
```
- **Expected Runtime**: ~20 seconds.
- **Output**: `D:\defect_cache\results\phase_3_vram.json`
- **GATE (Paste back to chat)**:
  1. `pytest` pass summary for `test_defect_synthetic.py`.
  2. Contents of `D:\defect_cache\results\phase_3_summary.json` and `D:\defect_cache\results\phase_3_vram.json` (confirming $K=16$ image AUROC $\ge 0.94$ and pixel AUROC $\ge 0.95$ on the 10 held-out MVTec categories).
- **On Failure**: Inspect `D:\defect_cache\logs\step3_fewshot.log` and `step3_cache_dino.log`.

---

## 3. Phase 4 — Stage 0/1 & Stage A (QC, Alignment, ROI Assembly Verification)

### Step 4.1: Run Unit & Leakage Tests
```powershell
git checkout v3/phase-4-stage-a
pytest -p no:cacheprovider v3/tests/test_align.py v3/tests/test_mismatch_synthetic.py v3/tests/test_leakage.py -v
```
- **Expected Runtime**: ~5–15 seconds.

### Step 4.2: Generate Normal-Train ROI Overlays for Human Recipe Review
```powershell
python -m v3.eval.roi_overlay --dry-run
python -m v3.eval.roi_overlay
```
- **Expected Runtime**: ~10 seconds.
- **Output**: Normal-train overlay PNGs in `D:\defect_cache\results\roi_overlays\`. Review overlays and update `status: reviewed` in `v3/config/recipes/*.yaml` once verified.

### Step 4.3: Evaluate Stage A vs. Stage B Only on MVTec LOCO (Logical vs. Structural Split)
```powershell
python -m v3.eval.step4_stage_a_eval --dry-run

$proc = Start-Process -FilePath "D:\envs\anomaly_v3\Scripts\python.exe" `
  -ArgumentList "-m v3.eval.step4_stage_a_eval --k 4 --seed 0" `
  -WorkingDirectory (Get-Location) -NoNewWindow -PassThru
Get-Content "D:\defect_cache\logs\step4_stage_a_eval.log" -Wait -Tail 20
```
- **Expected Runtime**: ~5–12 minutes (requires `D:\data\mvtec_loco`).
- **Log File**: `D:\defect_cache\logs\step4_stage_a_eval.log`
- **Output**: `D:\defect_cache\results\phase_4_loco_summary.json` and `D:\defect_cache\results\phase_4_loco.csv`
- **GATE (Paste back to chat)**:
  1. `pytest` output for `test_align.py`, `test_mismatch_synthetic.py`, and `test_leakage.py`.
  2. Contents of `D:\defect_cache\results\phase_4_loco_summary.json` (showing logical vs. structural AUROC delta over Stage-B-only).
- **On Failure**: If `D:\data\mvtec_loco` is not yet downloaded, `--dry-run` will report `loco_root_exists=False`. Place MVTec LOCO at `D:\data\mvtec_loco` and re-run.

---

## 4. Phase 5 — Fusion, Calibration, End-to-End Pipeline & Settings Freeze

### Step 5.1: Run End-to-End Pipeline & Contract Unit Tests
```powershell
git checkout v3/phase-5-fusion
pytest -p no:cacheprovider v3/tests/test_phase5_pipeline.py -v
```

### Step 5.2: Run 1080p VRAM & Determinism Gate (`< 7.0 GB` on RTX 4060 8 GB)
```powershell
python -m v3.eval.step5_memtest --dry-run
python -m v3.eval.step5_memtest
```
- **Expected Runtime**: ~15–30 seconds.
- **Log File**: `D:\defect_cache\logs\step5_memtest.log`
- **Output**: `D:\defect_cache\results\phase_5_memtest.json` (verifies `peak_vram_gb < 7.0` on a $1920 \times 1080$ image and exact bitwise/float determinism across 2 runs).

### Step 5.3: Run 4-Combination Comparison Table (B1, B2, B1+B2, A+B1+B2)
```powershell
python -m v3.eval.step5_compare --dry-run

$proc = Start-Process -FilePath "D:\envs\anomaly_v3\Scripts\python.exe" `
  -ArgumentList "-m v3.eval.step5_compare --k 4" `
  -WorkingDirectory (Get-Location) -NoNewWindow -PassThru
Get-Content "D:\defect_cache\logs\step5_compare.log" -Wait -Tail 20
```
- **Expected Runtime**: ~2–5 minutes on cached CLIP + DINOv2 features.
- **Log File**: `D:\defect_cache\logs\step5_compare.log`
- **Output**: `D:\defect_cache\results\phase_5_comparison.csv` and `D:\defect_cache\results\phase_5_summary.json`

### Step 5.4: Freeze Validation Settings + git SHA (`v3/config/frozen_settings.json`)
```powershell
python -m v3.eval.step5_freeze --dry-run
python -m v3.eval.step5_freeze
```
- **Expected Runtime**: < 5 seconds.
- **Output**: Writes `v3/config/frozen_settings.json` with git SHA, timestamp, frozen weights, and validation metrics.
- **GATE (Paste back to chat)**:
  1. `pytest` output for `test_phase5_pipeline.py`.
  2. Contents of `D:\defect_cache\results\phase_5_memtest.json`, `D:\defect_cache\results\phase_5_summary.json`, and `v3/config/frozen_settings.json`.
- **On Failure**: Check `D:\defect_cache\logs\step5_memtest.log` or `step5_compare.log`.

---

## 5. Phase 6 — APD Preparation & Evaluation Protocol

### Step 6.1: Run APD Protocol & Split Isolation Unit Tests
```powershell
git checkout v3/phase-6-apd
pytest -p no:cacheprovider v3/tests/ -v
```
- **Expected Runtime**: ~10–25 seconds for the full synthetic test suite.

### Step 6.2: Draw ROIs on Normal Reference Images Only (`draw_rois.py`)
Do **not** open any defective test image while defining ROIs. Run the interactive ROI drawing tool on 1–4 normal reference images from `D:\data\apd`:

```powershell
python -m v3.eval.draw_rois --dry-run
python -m v3.eval.draw_rois --image "D:\data\apd\normal\ref_000.png" --recipe "v3/config/recipes/apd_panel.yaml"
```
- Once all panel slots/zones are drawn from normal reference images and engineering knowledge, update `status: frozen` in `v3/config/recipes/apd_panel.yaml`.

### Step 6.3: Run Post-Freeze ROI Coverage Check (ONLY After ROIs Are Frozen)
Run this check **only after** `v3/config/recipes/apd_panel.yaml` is frozen. As required by `BUILD_PROMPT.md`, any coverage gap $< 50\%$ is logged as a limitation in `v3/docs/results/phase_6.md`—never redraw ROIs after inspecting defective mask coverage:

```powershell
python -m v3.eval.roi_coverage_check --dry-run
python -m v3.eval.roi_coverage_check --apd-root "D:\data\apd" --recipe "v3/config/recipes/apd_panel.yaml"
```
- **Output**: `D:\defect_cache\results\phase_6_roi_coverage.json`

### Step 6.4: Run Single-Shot Uncalibrated & Few-Shot APD Evaluation
```powershell
python -m v3.eval.step6_apd_protocol --dry-run

$proc = Start-Process -FilePath "D:\envs\anomaly_v3\Scripts\python.exe" `
  -ArgumentList "-m v3.eval.step6_apd_protocol --apd-root D:\data\apd --recipe v3/config/recipes/apd_panel.yaml" `
  -WorkingDirectory (Get-Location) -NoNewWindow -PassThru
Get-Content "D:\defect_cache\logs\step6_apd_protocol.log" -Wait -Tail 20
```
- **Expected Runtime**: ~3–10 minutes depending on APD image count.
- **Log File**: `D:\defect_cache\logs\step6_apd_protocol.log`
- **Output**: `D:\defect_cache\results\phase_6_apd_results.json` and `D:\defect_cache\results\phase_6_apd_per_class.csv`
- **GATE (Paste back to chat)**:
  1. `pytest` summary for `v3/tests/`.
  2. Contents of `D:\defect_cache\results\phase_6_roi_coverage.json` and `D:\defect_cache\results\phase_6_apd_results.json`.
- **On Failure**: Check `D:\defect_cache\logs\step6_apd_protocol.log` (verifies `v3/config/frozen_settings.json` exists and `apd_panel.yaml` has `status: frozen`).

