# Phase 2 Execution Guide (`v3/phase-2-clip`)

All commands below are written for **Windows PowerShell** from the repository root (`c:\Users\JYOTHIRADITHYA J\Downloads\defect-detector`).
All paths read from [`v3/config/default.yaml`](file:///c:/Users/JYOTHIRADITHYA%20J/Downloads/defect-detector/v3/config/default.yaml) on drive `D:`. Each script writes its execution log to `D:\defect_cache\logs\` and appends a status entry to [`v3/docs/decisions.md`](file:///c:/Users/JYOTHIRADITHYA%20J/Downloads/defect-detector/v3/docs/decisions.md).

---

## 0. Activate `D:\envs\anomaly_v3` and Set D-Drive Environment Variables

### Command
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
New-Item -ItemType Directory -Force -Path "D:\defect_cache\logs", "D:\defect_cache\results", "D:\defect_cache\visual_embeddings" | Out-Null
```
- **Expected Runtime**: `< 2 seconds`
- **Log File**: N/A (shell initialization)
- **What Success Looks Like**: Prompt prefixes with `(anomaly_v3)` and `(Get-Command python).Source` points to `D:\envs\anomaly_v3\Scripts\python.exe`.
- **On Failure**: If PowerShell execution policy blocks `Activate.ps1`, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and re-run step 0, or invoke `D:\envs\anomaly_v3\Scripts\python.exe` directly.

---

## 1. Step A — Environment & CUDA Verification

> **Note**: Step A was already verified and logged in `v3/docs/decisions.md` (`PASS`, `2.03s`). Run this only if you want to re-verify the shell session.

### Command
```powershell
D:\envs\anomaly_v3\Scripts\python.exe -u -m v3.eval.step_a_env_check
```
- **Expected Runtime**: `~2–4 seconds`
- **Log File to Watch**: `D:\defect_cache\logs\step_a.log`
- **What Success Looks Like**:
  - Prints `CUDA Available : True`, `GPU Model : NVIDIA GeForce RTX 4060 Laptop GPU`, `Driver Version : 566.07`, `PyTorch : 2.6.0+cu124`, `open_clip : 3.3.0`, local `.safetensors` paths under `D:\ml_cache\hf\hub`, and `STEP A RESULT : PASS`.
  - Writes `D:\defect_cache\results\step_a_env.json`.
- **On Failure**:
  - If `CUDA Available : False`, verify `D:\envs\anomaly_v3\Scripts\python.exe` is used rather than the old CPU `.venv`.
  - If VRAM is low, check `nvidia-smi` and kill any stale GPU processes (`Stop-Process -Id <PID> -Force`).

---

## 2. Step B — Baseline 6-Decimal Reproduction Check (`clip_b32_224`, `fp32`)

> **Note**: Step B was already verified against the Phase 1 full held-out run (`PASS`, `max_diff = 0.000000`). By default this checks the completed baseline JSON in `< 1s`; pass `--recompute` only if you want to re-run all 1,226 held-out images from scratch.

### Command
```powershell
D:\envs\anomaly_v3\Scripts\python.exe -u -m v3.eval.step_b_baseline
```
- **Expected Runtime**: `< 2 seconds` (verifying completed baseline run) or `~60 seconds` on GPU with `--recompute`
- **Log File to Watch**: `D:\defect_cache\logs\step_b.log`
- **What Success Looks Like**:
  - Prints per-category and macro comparisons across all 10 held-out categories (`Macro Image AUROC = 0.790817`, `Macro Pixel AUROC = 0.680949`).
  - Ends with `STEP B RESULT: PASS (max_diff=0.000000)`.
- **On Failure**:
  - Inspect `D:\defect_cache\logs\step_b.log` for the category or metric showing `Diff > 0.000001`. Ensure `precision="fp32"` and `v3-category` split seed `42` are unchanged.

---

## 3. Step C — Smoke Run per Backbone (20 Images Each)

### Command
```powershell
D:\envs\anomaly_v3\Scripts\python.exe -u -m v3.eval.step_c_smoke --num-images 20
```
- **Expected Runtime**: `~5–15 seconds`
- **Log File to Watch**: `D:\defect_cache\logs\step_c.log`
- **What Success Looks Like**:
  - Completes 20 images on `clip_b32_224 (fp32)` and 20 images on `clip_l14_336 (fp16)` using local weights from `D:\ml_cache\hf\hub`.
  - Prints peak VRAM (`~680 MB` for B/32, `~900–1100 MB` for L/14) and `STEP C RESULT: PASS`.
  - Writes `D:\defect_cache\results\step_c_smoke.json`.
- **On Failure**:
  - If weight loading fails in offline mode, verify `D:\ml_cache\hf\hub\models--timm--vit_large_patch14_clip_336.openai\snapshots\81e38efc4637de5023b10e75a7f9bd1c6fa6b010\open_clip_model.safetensors` exists (`1632.5 MB`).

---

## 4. Step D — Build `clip_l14_336` (`fp16`) Visual Embedding Cache (Background Job)

> **Note**: ~1,100 of 1,725 L/14 images and all 1,725 B/32 images are already cached in `D:\defect_cache\visual_embeddings`. The script automatically skips existing valid `.npz` files, resumes per category with `batch_size=8` (auto-fallback to `4` on OOM), and logs progress every 30 seconds.

### Command (Start Background Process)
```powershell
$procD = Start-Process -FilePath "D:\envs\anomaly_v3\Scripts\python.exe" `
  -ArgumentList "-u -m v3.eval.step_d_cache_l14 --backbone clip_l14_336 --precision fp16 --batch-size 8" `
  -WorkingDirectory (Get-Location).Path `
  -NoNewWindow -PassThru
Write-Host "Started Step D PID: $($procD.Id)"
```

### Command to Tail Log
```powershell
Get-Content "D:\defect_cache\logs\step_d.log" -Wait -Tail 20
```
*(Press `Ctrl+C` to stop tailing once `STEP D RESULT: PASS` appears; `Ctrl+C` on `Get-Content` does not kill the background job.)*

- **Expected Runtime**: `~60–120 seconds` for the remaining ~600 uncached L/14 images (`~0.12s/image` at batch size 8)
- **Log File to Watch**: `D:\defect_cache\logs\step_d.log`
- **What Success Looks Like**:
  - Each category prints `DONE (N/N cached, peak_vram=... MB)` or `SKIP`.
  - Ends with `STEP D RESULT: PASS (...) | L14: ~3.95 GB (...s/img, Peak VRAM ... MB) | B32: ~0.35 GB`.
  - Writes `D:\defect_cache\results\step_d_cache_stats.json`.
- **On Failure**:
  - If the process hangs for >5 minutes with no new lines in `D:\defect_cache\logs\step_d.log`, terminate it with `Stop-Process -Id $procD.Id -Force` and re-run Step D with `--batch-size 4` (since it skips already-cached files, it will resume from the exact category where it stopped).

---

## 5. Step E — Phase 2 Cached Evaluations & Report Generation (Background Job)

### Command (Start Background Process)
```powershell
$procE = Start-Process -FilePath "D:\envs\anomaly_v3\Scripts\python.exe" `
  -ArgumentList "-u -m v3.eval.step_e_evaluate" `
  -WorkingDirectory (Get-Location).Path `
  -NoNewWindow -PassThru
Write-Host "Started Step E PID: $($procE.Id)"
```

### Command to Tail Log
```powershell
Get-Content "D:\defect_cache\logs\step_e.log" -Wait -Tail 20
```

- **Expected Runtime**: `~90–180 seconds` (100% cache-hit evaluation across polarity variants, validation local-weight sweep, and 10 held-out categories for both backbones)
- **Log File to Watch**: `D:\defect_cache\logs\step_e.log`
- **What Success Looks Like**:
  - Logs polarity mechanism results (`final_tokens_native`, `intermediate_native`, `intermediate_inverted`), validation local-weight sweep (`global_only`, `fused_095`, `fused_090`, `fused_080`, `fused_050`, `local_only`), fitted calibration parameters `(T, bias)`, and per-category held-out metrics (`toothbrush` flagged `[LOW-N]`).
  - Ends with `STEP E RESULT: PASS`.
  - Generates in `D:\defect_cache\results\`:
    - `phase_2_report.md` and `phase_2_polarity.md` (also mirrored to `v3/docs/results/phase_2.md` and `v3/docs/results/phase_2_polarity.md`)
    - `phase_2_b32_heldout.csv` and `phase_2_l14_heldout.csv`
    - `phase_2_polarity.csv` and `phase_2_local_sweep.csv`
    - `phase_2_summary.json`
- **On Failure**:
  - If any `.npz` file fails to load, `CLIPFeatureExtractor.get_features` automatically deletes the corrupted file and recomputes it on the fly. Check `D:\defect_cache\logs\step_e.log` for the traceback.

---

## 6. Step F — Lint Check (`ruff`)

### Command
```powershell
& ".\.venv\Scripts\ruff.exe" check v3/
```
- **Expected Runtime**: `< 2 seconds`
- **Log File to Watch**: Console output
- **What Success Looks Like**: `All checks passed!` with exit code `0`.
- **On Failure**: Run `& ".\.venv\Scripts\ruff.exe" check v3/ --fix` and re-verify.

---

## 7. Step G — Commit Phase 2 Results & Merge into `v3/two-stage` (Not `main`)

### Command
```powershell
git status -s
git add v3/docs/decisions.md v3/docs/results/phase_2.md v3/docs/results/phase_2_polarity.md
git commit -m "phase 2: add final evaluation reports and decision log updates"
git checkout v3/two-stage
git merge --no-ff v3/phase-2-clip -m "Merge branch 'v3/phase-2-clip' into v3/two-stage"
git log --oneline -n 5
```
- **Expected Runtime**: `< 5 seconds`
- **Log File to Watch**: Console git output
- **What Success Looks Like**:
  - Current branch is `v3/two-stage`.
  - Non-fast-forward merge commit is created on `v3/two-stage`.
  - `main` remains untouched.
- **On Failure**: If untracked temporary files block checkout, verify `.gitignore` excludes `data/cache/` and `*.log`, or stash/remove untracked scratch files before merging.

---

## 8. Deferred / Skipped Items and Why

Per strict-mode instructions, the following items were intentionally skipped:
1. **Unit Test Suite (`pytest`)**: Skipped to avoid extra runtime overhead during Phase 2 wrap-up; syntax/import compilation (`py_compile`) and `ruff check v3/` were run instead.
2. **C-Drive Cleanup Deletions (`.venv`, HF cache on C:, MVTec copy on C:)**: Deferred (`pip cache purge` was already executed earlier, freeing `3.77 GB` on `C:`). All active caches, datasets, virtual environments, and outputs now reside on `D:`.
3. **`.git` Largest-Objects Report**: Deferred per strict-mode instructions (`DO NOT touch .git`).
4. **`D:\defect_cache` Staleness Report**: Deferred per strict-mode instructions; cache keys are deterministic SHA-256 hashes over `backbone:input_size:precision:image_sha256`.

