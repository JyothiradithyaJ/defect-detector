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
