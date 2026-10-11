# PHASE 6: EV Body Panel Track on APD — STATUS: CODE COMPLETE (PENDING USER GATE RUN)

## Task List
- [x] Update `v3/config/recipes/apd_panel.yaml` with README section 6 skeleton, README section 2.5 category-to-stage mapping, `[Published]` reference points with the `224 px` caveat, and `status: needs-human-roi-drawing`.
- [x] Implement `v3/eval/draw_rois.py` (interactive OpenCV polygon drawing tool on the golden template; never run automatically).
- [x] Implement `v3/eval/step6_apd_protocol.py` (README 8.3 protocol: refuses to run unless `frozen_settings.json` + `.sha256` match; read-only check on `D:\data\apd`; $k$ references by seed removed from test negatives; ROI file hash immutability assertion; overall, per-category, per-severity 0/1/2, Stage B only vs Stage A only vs A+B; optional `--two-fold-retune` flag excluding references and preserving ROI geometry).
- [x] Implement `v3/eval/roi_coverage_check.py` (read-only defect-mask overlap test; run by user only after ROIs are frozen).
- [x] Implement unit tests in `v3/tests/test_phase6_apd.py`.

## Built
- [`v3/config/recipes/apd_panel.yaml`](file:///c:/Users/JYOTHIRADITHYA%20J/Downloads/defect-detector/v3/config/recipes/apd_panel.yaml)
- [`v3/eval/draw_rois.py`](file:///c:/Users/JYOTHIRADITHYA%20J/Downloads/defect-detector/v3/eval/draw_rois.py)
- [`v3/eval/step6_apd_protocol.py`](file:///c:/Users/JYOTHIRADITHYA%20J/Downloads/defect-detector/v3/eval/step6_apd_protocol.py)
- [`v3/eval/roi_coverage_check.py`](file:///c:/Users/JYOTHIRADITHYA%20J/Downloads/defect-detector/v3/eval/roi_coverage_check.py)
- [`v3/tests/test_phase6_apd.py`](file:///c:/Users/JYOTHIRADITHYA%20J/Downloads/defect-detector/v3/tests/test_phase6_apd.py)

## Exact Reproduce Commands
```powershell
# 1. After placing APD at D:\data\apd (following license acceptance), draw ROIs on golden template:
D:\envs\anomaly_v3\Scripts\python.exe -u -m v3.eval.draw_rois --template D:/data/apd/normal/000.png

# 2. Run frozen APD protocol (requires v3/config/frozen_settings.json from Phase 5):
D:\envs\anomaly_v3\Scripts\python.exe -u -m v3.eval.step6_apd_protocol

# 3. Run read-only ROI coverage check ONLY after ROIs are frozen:
D:\envs\anomaly_v3\Scripts\python.exe -u -m v3.eval.roi_coverage_check
```

## Published Reference Points (`[Published]`, README Section 2.3)
> **Note**: `[Published]` numbers below are cited directly from the APD benchmark literature where baselines were evaluated at **`224 px`** input resolution (not re-implemented).

| Method | Source | Input Resolution | Image AUROC | Pixel AU-PRO |
|---|---|---:|---:|---:|
| PatchCore | `[Published]` | 224 px | 0.785 | 0.642 |
| EfficientAD | `[Published]` | 224 px | 0.761 | 0.618 |
| WinCLIP | `[Published]` | 224 px | 0.712 | 0.534 |
| **v3 Stage B Only (DINO + CLIP)** | Ours (560 px / 336 px) | 560 px | `pending, run by user` | `pending, run by user` |
| **v3 Stage A Only (ROIs)** | Ours (1920x1080) | 1080p | `pending, run by user` | `pending, run by user` |
| **v3 Fused (Stage A + B)** | Ours (1080p + 560 px) | 1080p / 560 px | `pending, run by user` | `pending, run by user` |

## Limitations (README Section 2.4)
1. **Viewpoint & Alignment**: Stage A ROI inspection requires a fixed camera viewpoint or a convergent Stage 1 registration onto the golden template; severe 3D viewpoint changes or occluded fiducials trigger `misaligned` / `RECAPTURE`.
2. **Recipe-Bound Logical Rules**: Stage A logical detectors (`presence`, `count`, `order`, `position`) only inspect polygons defined in the human-authored recipe YAML and do not infer unannotated assembly constraints.
3. **Spatial Resolution Floor**: Stage B DINOv2 patch matching detects fine surface scratches and dents (~3–5 px at 560 px short side) but cannot resolve sub-pixel micro-cracks without higher-resolution tiling.
4. **Calibration Dependency**: Probability outputs require logistic calibration on labelled validation data (`calibrated, not strictly zero-shot`).

## Gate & Verification Status
- **`v3/tests/test_phase6_apd.py`**: `PASS` (verified `frozen_settings.json` SHA-256 enforcement, 2-fold retune reference exclusion, ROI coverage diagnostic, and APD recipe schema).
- **Frozen Settings Verified**: [`v3/config/frozen_settings.json`](file:///c:/Users/JYOTHIRADITHYA%20J/Downloads/defect-detector/v3/config/frozen_settings.json) (`SHA-256 = 153e12546c82333b863b255349e6d2644805b97fb1444a4676f47ea5da9dc70b`).
- **APD Dataset Check (`step6_apd_protocol.py --dry-run`)**: `D:\data\apd` (`exists=False`). Per `BUILD_PROMPT.md` Hard Rule 4, no automatic download or URL guessing is performed; once the APD license is accepted and images are placed in `D:\data\apd` with ROIs drawn via `draw_rois.py`, re-run `step6_apd_protocol.py` and `roi_coverage_check.py`.

