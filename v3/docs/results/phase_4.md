# PHASE 4: Stage A (Quality Check, Alignment & ROI Mismatch Engine) — STATUS: CODE COMPLETE (PENDING USER GATE RUN)

## Task List
- [x] Implement `v3/core/qc.py` with normal median/MAD thresholds (`blur_floor_percentile` from config) and `RECAPTURE`-only failure verdicts.
- [x] Implement `v3/core/align.py` with rigid registration restricted to `recipe.stable_roi`, diverged `misaligned` hard mismatch (skipping positional checks), and DINO CLS `wrong_variant` check.
- [x] Implement `v3/core/roi.py` and detectors `presence.py`, `count.py`, `order.py`, `position.py` with leave-one-out normal median/MAD calibration (`decide.min_mad` floor) and `early_exit` support.
- [x] Author MVTec LOCO draft recipes (`breakfast_box.yaml`, `juice_bottle.yaml`, `pushpins.yaml`, `screw_bag.yaml`, `splicing_connectors.yaml`) marked `status: draft-needs-human-review`.
- [x] Implement `v3/eval/roi_overlay.py` (with commit/seed/config stamp) and `v3/eval/step4_stage_a_eval.py`.
- [x] Implement unit tests in `v3/tests/test_align.py`, `v3/tests/test_mismatch_synthetic.py`, and `v3/tests/test_leakage.py`.

## Built
- [`v3/core/qc.py`](file:///c:/Users/JYOTHIRADITHYA%20J/Downloads/defect-detector/v3/core/qc.py): `QualityChecker` (`S0`).
- [`v3/core/align.py`](file:///c:/Users/JYOTHIRADITHYA%20J/Downloads/defect-detector/v3/core/align.py): `Stage1Aligner` (`S1`).
- [`v3/core/roi.py`](file:///c:/Users/JYOTHIRADITHYA%20J/Downloads/defect-detector/v3/core/roi.py) & `v3/core/mismatch/{presence,count,order,position}.py`: `StageAROIEngine` (`S2`).
- `v3/config/recipes/{breakfast_box,juice_bottle,pushpins,screw_bag,splicing_connectors}.yaml`.
- [`v3/eval/roi_overlay.py`](file:///c:/Users/JYOTHIRADITHYA%20J/Downloads/defect-detector/v3/eval/roi_overlay.py) and [`v3/eval/step4_stage_a_eval.py`](file:///c:/Users/JYOTHIRADITHYA%20J/Downloads/defect-detector/v3/eval/step4_stage_a_eval.py).

## Exact Reproduce Commands
```powershell
D:\envs\anomaly_v3\Scripts\python.exe -u -m v3.eval.roi_overlay --dry-run
D:\envs\anomaly_v3\Scripts\python.exe -u -m v3.eval.step4_stage_a_eval
```

## Reproducibility Stamp
- **Branch**: `v3/phase-4-stage-a`
- **Seeds**: `[0, 1, 2]`
- **Config Path**: `v3/config/default.yaml`

## Gate to Check
- Paste the contents of `D:\defect_cache\results\phase_4_stage_a.json` (per-kind precision/recall across `presence`, `count`, `order`, `position` and normal false-reject rate vs `0.05`).

## Numbers
- **Status**: `pending, run by user` (see `RUN_PHASES.md` Phase 4 section).

## Open Issues
- MVTec LOCO recipe YAMLs are marked `status: draft-needs-human-review`; review rendered overlays via `v3.eval.roi_overlay` before running full LOCO benchmarks.

