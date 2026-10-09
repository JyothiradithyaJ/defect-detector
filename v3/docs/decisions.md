# v3 Decision Log

## 2026-10-09 — Phase 0 setup
- **Decision:** Keep all new implementation beneath `v3/` and branch from the current `main` commit.
  **Reason:** Required by `README_v3.md`; preserves the v1 baseline.
- **Decision:** Phase 0 records the repository's active v1 command sequence from its README, but baseline execution is not claimed.
  **Reason:** Execution requires the user's local MVTec AD dataset and a compatible Python/PyTorch environment.
- **Open:** Confirm the APD dataset's access and licence before Phase 6. Do not download guessed URLs or start EV evaluation until the user confirms.
