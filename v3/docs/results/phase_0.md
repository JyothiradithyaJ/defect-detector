# Phase 0 — Setup and freeze

Status: **BLOCKED — baseline reproducibility not yet verified**

## Built
- Created branch `v3/two-stage` from `main`.
- Added the initial v3 configuration under `v3/config/default.yaml`.
- Added the decision log at `v3/docs/decisions.md`.
- Recorded v1 reference commit: `232f9d3fe64852e7b64dc178064907cf28080072`.

## Existing v1 run sequence
From the repository root, with Python dependencies installed and MVTec AD extracted to `data/mvtec_ad/`:
```bash
python backend/scripts/create_mvtec_manifests.py
python backend/scripts/fit_temperature.py --device cpu
python backend/scripts/evaluate_mvtec.py --device cpu
```
The README documents these commands. Dataset presence, environment, and successful execution have not been verified in this remote-only setup.

## Gate
- Branch and v3 starter files: **PASS**
- Pytest execution: **NOT RUN** (no repository runtime is available through the GitHub file connector)
- v1 command documented: **PASS**
- v1 command reproducible and v1 evaluation recorded: **BLOCKED**, pending execution in an environment with MVTec AD.

## Reproducibility metadata
- Base commit: `232f9d3fe64852e7b64dc178064907cf28080072`
- Seed/config hash/result metrics: **not recorded**; no evaluation run occurred.
- No performance results are asserted by this file.

## Open issues
- Run `pytest` on the local checkout after installing `backend/requirements.txt`.
- Confirm the MVTec AD data path and run the v1 sequence above; attach the raw results and environment details before Phase 1.
- APD access and licence must be confirmed by the user before Phase 6.

## Next
Phase 1 (evaluation harness and v1 baseline) is proposed but not started. Awaiting the user's go-ahead and completion of the Phase 0 gate.
