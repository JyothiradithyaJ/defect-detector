# RUN_MANUAL.md — PowerShell Manual Execution Guide (Remaining Steps Only)

This guide lists **only the remaining manual steps** required to complete real-dataset Stage A validation, APD (`Assembled Panel Dataset for EV charging stations`) evaluation, and the single-image end-to-end demo on branch `v3/phase-6-apd`.

---

## 0. Prerequisites: Dataset Placement & License Acceptance

1. **APD (Assembled Panel Dataset for EV charging stations)**:
   - Accept the dataset license from the official APD authors/repository.
   - Place the extracted dataset at `D:\data\apd` (expected subfolders: `D:\data\apd\normal\`, `D:\data\apd\anomalous\` or category subfolders, and `D:\data\apd\ground_truth\`).
   - **Never** commit any APD image to git or run automatic download scripts.
2. **MVTec LOCO AD (Optional/Recommended for Real Stage A Re-Validation)**:
   - As noted in Part 1, `step4_stage_a_eval.py` only evaluated Stage A on 40 synthetic 256×256 perturbations. To validate Stage A on real industrial assembly images prior to APD, place MVTec LOCO at `D:\data\mvtec_loco`.
3. **Frozen Settings Check**:
   - `v3/config/frozen_settings.json` and `v3/config/frozen_settings.sha256` (`153e12546c82333b863b255349e6d2644805b97fb1444a4676f47ea5da9dc70b`) are already generated and committed. Do not re-run `step5_freeze.py` before APD evaluation.

---

## 1. Environment Setup Block (Run Once per PowerShell Session)

```powershell
Set-Location "<REPO_ROOT>"
& "D:\envs\anomaly_v3\Scripts\Activate.ps1"

$env:HF_HOME = "D:\ml_cache\hf"
$env:TORCH_HOME = "D:\ml_cache\torch"
$env:UV_CACHE_DIR = "D:\ml_cache\uv"
$env:PIP_CACHE_DIR = "D:\ml_cache\pip"
$env:TEMP = "D:\ml_cache\tmp"
$env:TMP = "D:\ml_cache\tmp"
$env:UV_LINK_MODE = "copy"
$env:HF_HUB_OFFLINE = "1"

git status -s
nvidia-smi
```

---

## 2. Step A: APD Placement & Freeze Prerequisite Check (`< 5 s`, CPU only)

- **Command**:
  ```powershell
  & "D:\envs\anomaly_v3\Scripts\python.exe" -m v3.eval.step6_apd_protocol --dry-run
  ```
- **Expected Runtime**: ~1 second.
- **Log File**: Console output (`[DRY-RUN] step6_apd_protocol.py: ...`).
- **Success Condition**: Prints `apd_root=D:\data\apd (exists=True)` and `frozen_settings=...\v3\config\frozen_settings.json (exists=True)`.
- **Exact Numbers / Output to Paste Back**: The one-line `[DRY-RUN]` console output confirming `exists=True` for both paths.
- **On Failure**: If `apd_root=D:\data\apd (exists=False)`, verify the folder path `D:\data\apd` exists and is readable.

---

## 3. Step B: Hand-Draw & Freeze ROIs on Golden Normal Template (`Interactive`, CPU only)

> **Strict Rule**: Open **only** normal reference images (`D:\data\apd\normal\...`). Never open or inspect defective/anomalous test images while drawing or adjusting ROIs.

- **Command**:
  ```powershell
  & "D:\envs\anomaly_v3\Scripts\python.exe" -m v3.eval.draw_rois --dry-run
  & "D:\envs\anomaly_v3\Scripts\python.exe" -m v3.eval.draw_rois --template "D:\data\apd\normal\golden_template.png" --recipe "v3/config/recipes/apd_panel.yaml"
  ```
  *(After drawing polygons for `stable_roi`, `ev_port_left_cables`, `sensor_right_wires`, `red_cap_left`, and `pcb_left`, set `status: human-reviewed-and-frozen` and `expected_count` / `expected_sequence` in `v3/config/recipes/apd_panel.yaml`, then render the verification overlay)*:
  ```powershell
  & "D:\envs\anomaly_v3\Scripts\python.exe" -m v3.eval.roi_overlay --recipe "v3/config/recipes/apd_panel.yaml"
  ```
- **Expected Runtime**: ~2–5 minutes (human interactive drawing) + ~2 seconds for `roi_overlay`.
- **Log / Output File**: `D:\defect_cache\results\roi_overlays\apd_panel_rois.png`.
- **Success Condition**: `v3/config/recipes/apd_panel.yaml` has `status: human-reviewed-and-frozen`, non-empty `polygon` coordinates, and `apd_panel_rois.png` visually aligns with the normal template.
- **Exact Numbers / Output to Paste Back**: Paste the updated `v3/config/recipes/apd_panel.yaml` and the SHA-256 printed by `Get-FileHash v3/config/recipes/apd_panel.yaml -Algorithm SHA256`.
- **On Failure**: If OpenCV window does not open, edit polygon coordinates directly in `v3/config/recipes/apd_panel.yaml` from normal template pixel coordinates and verify alignment using `python -m v3.eval.roi_overlay --recipe v3/config/recipes/apd_panel.yaml`.

---

## 4. Step C: Stage A Re-Validation on Real Normal/Logical Images (`~2–5 min`, Background Job)

Because `step4_stage_a_eval.py` only tested synthetic 256×256 boards in memory, verify Stage A false-reject rate and ROI z-scores on real normal reference images (and `D:\data\mvtec_loco` recipes if placed) before/alongside APD:

- **Command**:
  ```powershell
  & "D:\envs\anomaly_v3\Scripts\python.exe" -m v3.eval.roi_overlay
  $job = Start-Process -FilePath "D:\envs\anomaly_v3\Scripts\python.exe" `
    -ArgumentList "-m v3.eval.step4_stage_a_eval" `
    -WorkingDirectory "<REPO_ROOT>" `
    -NoNewWindow -PassThru
  Get-Content "D:\defect_cache\logs\step4_stage_a_eval.log" -Wait -Tail 20
  ```
- **Expected Runtime**: ~7 seconds (synthetic baseline) or ~2–5 minutes when run against real LOCO recipes.
- **Log File to Watch**: `D:\defect_cache\logs\step4_stage_a_eval.log` (press `Ctrl+C` to stop tailing once `STEP 4 RESULT: PASS` appears).
- **Success Condition**: `Normal False-Reject Rate <= 0.0500` and `Recipe YAML immutability check: PASS`.
- **Exact Numbers to Paste Back**: Contents of `D:\defect_cache\results\phase_4_stage_a.json`.
- **On Failure**: Check `D:\defect_cache\logs\step4_stage_a_eval.log`; if recipe hash mismatch is raised, restore `git checkout -- v3/config/recipes/`.

---

## 5. Step D: Run Frozen Phase 6 APD Protocol (`~3–8 min`, Uses GPU — Background Job)

- **Command**:
  ```powershell
  $job = Start-Process -FilePath "D:\envs\anomaly_v3\Scripts\python.exe" `
    -ArgumentList "-m v3.eval.step6_apd_protocol" `
    -WorkingDirectory "<REPO_ROOT>" `
    -NoNewWindow -PassThru
  Get-Content "D:\defect_cache\logs\step6_apd_protocol.log" -Wait -Tail 20
  ```
- **Expected Runtime**: ~3–8 minutes (depending on uncached APD images; uses GPU ~630 MB VRAM).
- **Log File to Watch**: `D:\defect_cache\logs\step6_apd_protocol.log` (press `Ctrl+C` to exit `Get-Content -Wait` after `STEP 6 RESULT: PASS`).
- **Success Condition**: Log ends with `STEP 6 RESULT: PASS` and writes `D:\defect_cache\results\phase_6_apd_summary.json`.
- **Exact Numbers to Paste Back**: Full contents of `D:\defect_cache\results\phase_6_apd_summary.json` (overall, per-category, per-severity 0/1/2, Stage B only vs Stage A only vs A+B, and `recipe_sha256`).
- **On Failure**:
  - If `Frozen settings SHA-256 mismatch`: check `git status v3/config/frozen_settings.*`.
  - If `status='needs-human-roi-drawing'`: complete Step B first.

---

## 6. Step E: Post-Freeze Read-Only ROI Coverage Check (`~10–30 s`, CPU only)

> **Strict Rule**: Run **only after** Step B (`status: human-reviewed-and-frozen`) and Step D are complete. Never modify `apd_panel.yaml` after running this script.

- **Command**:
  ```powershell
  & "D:\envs\anomaly_v3\Scripts\python.exe" -m v3.eval.roi_coverage_check --dry-run
  & "D:\envs\anomaly_v3\Scripts\python.exe" -m v3.eval.roi_coverage_check --recipe "v3/config/recipes/apd_panel.yaml" --masks-dir "D:\data\apd\ground_truth"
  ```
- **Expected Runtime**: ~10–30 seconds.
- **Log / Output File**: `D:\defect_cache\results\roi_coverage.json`.
- **Success Condition**: Prints `ROI Coverage Check complete: mean_coverage=...` and verifies `apd_panel.yaml` SHA-256 is unchanged.
- **Exact Numbers to Paste Back**: Contents of `D:\defect_cache\results\roi_coverage.json` (`num_masks`, `mean_coverage`, `recipe_sha256`).
- **On Failure**: If `Ground-truth mask directory not found`, pass `--masks-dir` pointing to the APD binary mask folder.

---

## 7. Step F: Single-Image End-to-End Demo (`~5–15 s`, Uses GPU if available)

Run `v3/eval/demo_single_image.py` on one image (either an MVTec test image with an existing recipe or an APD panel image once `apd_panel.yaml` is frozen):

- **Command**:
  ```powershell
  & "D:\envs\anomaly_v3\Scripts\python.exe" -m v3.eval.demo_single_image `
    --image "D:\data\mvtec_ad\bottle\test\broken_large\000.png" `
    --recipe "v3/config/recipes/breakfast_box.yaml" `
    --out-dir "D:\defect_cache\results\demo_single" `
    --k 4 `
    --dry-run

  & "D:\envs\anomaly_v3\Scripts\python.exe" -m v3.eval.demo_single_image `
    --image "D:\data\mvtec_ad\bottle\test\broken_large\000.png" `
    --recipe "v3/config/recipes/breakfast_box.yaml" `
    --out-dir "D:\defect_cache\results\demo_single" `
    --k 4
  ```
- **Expected Runtime**: ~5–15 seconds (skips feature extraction if `.npz` already exists in `D:\defect_cache\dino_features`).
- **Output Files**: `D:\defect_cache\results\demo_single\result.json` and `D:\defect_cache\results\demo_single\overlay.png`.
- **Success Condition**: Exit code `0`; `result.json` contains `verdict`, `mismatches`, `image_score`, `timing_seconds`, `peak_vram_mb`, `commit_hash`, `seed`, `config_hash`, and `frozen_settings_sha256`; `overlay.png` exists.
- **Exact Numbers to Paste Back**: Full contents of `D:\defect_cache\results\demo_single\result.json`.
- **On Failure**: One-line `ERROR: <Type>: <Message>` is printed to stderr. If `No normal reference images found`, ensure `--image` is inside a dataset folder with `train/good/*.png` or `template_path` in `--recipe` points to a folder of normal images.

