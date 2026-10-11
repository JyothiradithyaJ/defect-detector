"""Step A: Environment and CUDA verification script.

Verifies torch.cuda.is_available() is True, reports GPU model, driver version,
free/total VRAM, torch and open_clip versions, and checks local checkpoint files on D:.
Writes log to D:/defect_cache/logs/step_a.log and appends status to v3/docs/decisions.md.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import open_clip
import torch

from v3.config import DEFAULT_CONFIG_PATH, StepLogger, load_v3_config, log_decision
from v3.core.features import BACKBONE_SPECS, resolve_local_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step A: Environment & CUDA check for Phase 2.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to v3 YAML config.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    cfg = load_v3_config(args.config)
    log_path = cfg.logs_dir / "step_a.log"

    with StepLogger(log_path) as logger:
        try:
            logger.info("=== STEP A: Environment & CUDA Verification ===")
            cuda_ok = bool(torch.cuda.is_available())
            if not cuda_ok:
                raise RuntimeError("torch.cuda.is_available() returned False. Ensure CUDA wheels are installed.")

            gpu_name = torch.cuda.get_device_name(0)
            free_vram, total_vram = torch.cuda.mem_get_info()
            free_gb = free_vram / (1024**3)
            total_gb = total_vram / (1024**3)

            try:
                driver = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    text=True,
                    timeout=10,
                ).strip()
            except (OSError, subprocess.SubprocessError):
                driver = "unknown"

            checkpoints = {
                backbone_id: resolve_local_checkpoint(backbone_id)
                for backbone_id in BACKBONE_SPECS
            }

            info = {
                "status": "PASS",
                "cuda_available": cuda_ok,
                "gpu": gpu_name,
                "driver": driver,
                "free_vram_gb": round(free_gb, 2),
                "total_vram_gb": round(total_gb, 2),
                "torch_version": torch.__version__,
                "open_clip_version": open_clip.__version__,
                "local_checkpoints": checkpoints,
                "data_root": str(cfg.data_root),
                "cache_dir": str(cfg.cache_dir),
                "results_dir": str(cfg.results_dir),
                "logs_dir": str(cfg.logs_dir),
            }

            logger.info(f"CUDA Available : {cuda_ok}")
            logger.info(f"GPU Model      : {gpu_name}")
            logger.info(f"Driver Version : {driver}")
            logger.info(f"Free VRAM      : {free_gb:.2f} GB / {total_gb:.2f} GB")
            logger.info(f"PyTorch        : {torch.__version__}")
            logger.info(f"open_clip      : {open_clip.__version__}")
            for b_id, ckpt in checkpoints.items():
                logger.info(f"Checkpoint [{b_id}]: {ckpt}")

            out_json = cfg.results_dir / "step_a_env.json"
            out_json.write_text(json.dumps(info, indent=2), encoding="utf-8")

            elapsed = time.perf_counter() - t0
            logger.info(f"STEP A RESULT  : PASS ({elapsed:.2f}s)")
            log_decision(
                cfg,
                "Step A (CUDA & Env Check)",
                "PASS",
                elapsed,
                f"GPU={gpu_name}, driver={driver}, free_vram={free_gb:.2f}GB, torch={torch.__version__}",
            )
            return 0
        except (RuntimeError, OSError, ValueError) as exc:
            elapsed = time.perf_counter() - t0
            logger.error(f"Step A failed: {exc}")
            log_decision(cfg, "Step A (CUDA & Env Check)", "FAIL", elapsed, str(exc))
            return 1


if __name__ == "__main__":
    sys.exit(main())

