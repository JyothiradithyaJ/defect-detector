"""Download DINOv2 ViT-L/14 + registers checkpoint into D:\\ml_cache\\torch if not present.

This is the ONLY script that touches the network (when HF_HUB_OFFLINE is unset).
Skips download if weights already exist in D:\\ml_cache\\torch and performs an offline load check.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from v3.config import DEFAULT_CONFIG_PATH, StepLogger, load_v3_config, log_decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and verify DINOv2 ViT-L/14 + registers weights.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to v3 YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned action and estimated size, then exit.")
    return parser.parse_args()


def find_existing_dino_checkpoint(torch_home: Path) -> Path | None:
    """Check D:\\ml_cache\\torch\\checkpoints for existing dinov2_vitl14_reg4_pretrain.pth."""
    ckpt_dir = torch_home / "checkpoints"
    if ckpt_dir.is_dir():
        for p in ckpt_dir.glob("*dinov2*vitl14*reg*.pth"):
            if p.is_file() and p.stat().st_size > 500_000_000:
                return p
    return None


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    cfg = load_v3_config(args.config)
    backbone = str(cfg.raw.get("dino", {}).get("model", "dinov2_vitl14_reg"))
    existing = find_existing_dino_checkpoint(cfg.torch_home)

    if args.dry_run:
        status = f"SKIP (already cached at {existing})" if existing else "DOWNLOAD (~1.21 GB)"
        print(
            f"[DRY-RUN] download_dino.py: model={backbone}, target_dir={cfg.torch_home}, "
            f"action={status}, estimated_output_size=1.21 GB"
        )
        return 0

    log_path = cfg.logs_dir / "step3_download_dino.log"
    with StepLogger(log_path) as logger:
        try:
            logger.info(f"=== Phase 3 Prerequisite: DINOv2 Checkpoint Verification ({backbone}) ===")
            torch.hub.set_dir(str(cfg.torch_home))
            if existing is not None:
                logger.info(f"Found existing checkpoint on D: {existing} ({existing.stat().st_size / (1024**2):.1f} MB) -> SKIP download.")
            else:
                if os.environ.get("HF_HUB_OFFLINE") == "1":
                    raise RuntimeError(
                        "DINOv2 checkpoint not found locally and HF_HUB_OFFLINE=1 is set. "
                        "Unset HF_HUB_OFFLINE (`Remove-Item Env:\\HF_HUB_OFFLINE`) for this download step only."
                    )
                logger.info(f"Downloading {backbone} into {cfg.torch_home}...")

            model = torch.hub.load("facebookresearch/dinov2", backbone, pretrained=True)
            model.eval()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            elapsed = time.perf_counter() - t0
            logger.info(f"DINOv2 offline load check: PASS ({elapsed:.2f}s)")
            log_decision(cfg, "Phase 3 DINOv2 Download & Check", "PASS", elapsed, f"model={backbone}")
            return 0
        except (RuntimeError, OSError, ValueError) as exc:
            elapsed = time.perf_counter() - t0
            logger.error(f"download_dino failed: {exc}")
            log_decision(cfg, "Phase 3 DINOv2 Download & Check", "FAIL", elapsed, str(exc))
            return 1


if __name__ == "__main__":
    sys.exit(main())
