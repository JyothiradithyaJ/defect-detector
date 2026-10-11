"""Step C: Smoke run per backbone (20 images each).

Runs a 20-image smoke test on CUDA for:
1. clip_b32_224 (fp32)
2. clip_l14_336 (fp16)
Logs peak VRAM and time per image to D:/defect_cache/logs/step_c.log and updates decisions.md.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import torch

from v3.config import DEFAULT_CONFIG_PATH, StepLogger, load_v3_config, log_decision
from v3.core.features import CLIPFeatureExtractor
from v3.eval.phase2_engine import Phase2Scorer
from v3.eval.splits import DatasetSample, create_v3_category_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step C: 20-image smoke run per CLIP backbone.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to v3 YAML config.")
    parser.add_argument("--num-images", type=int, default=20, help="Number of images per backbone smoke run.")
    return parser.parse_args()


def run_backbone_smoke(
    backbone_id: str,
    precision: str,
    samples: list[DatasetSample],
    cache_dir: Path,
    logger: StepLogger,
) -> dict:
    """Run smoke test on `samples` for a single backbone and report timing & peak VRAM."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    extractor = CLIPFeatureExtractor(
        backbone_id=backbone_id,
        device="cuda" if torch.cuda.is_available() else "cpu",
        precision=precision,
        cache_dir=cache_dir,
    )
    scorer = Phase2Scorer(
        extractor=extractor,
        vocabulary_mode="domain_generic",
        polarity_mode="final_tokens_native",
    )

    for idx, sample in enumerate(samples, 1):
        logit, prob, hmap, _ = scorer.score_sample(sample, metric_max_side=256, need_heatmap=True)
        if idx in (1, len(samples)):
            logger.info(
                f"  [{backbone_id} ({precision})] Image {idx}/{len(samples)} ({sample.category}): "
                f"logit={logit:.4f}, prob={prob:.4f}, heatmap={hmap.shape}"
            )

    elapsed = time.perf_counter() - t0
    sec_per_img = elapsed / max(1, len(samples))
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0

    logger.info(
        f"Smoke [{backbone_id} ({precision})]: {len(samples)} images in {elapsed:.2f}s "
        f"({sec_per_img:.4f}s/img) | Peak VRAM: {peak_vram_mb:.1f} MB | Source: {extractor.pretrained_source}"
    )

    del scorer
    del extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "backbone_id": backbone_id,
        "precision": precision,
        "num_images": len(samples),
        "elapsed_seconds": round(elapsed, 3),
        "seconds_per_image": round(sec_per_img, 4),
        "peak_vram_mb": round(peak_vram_mb, 1),
    }


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    cfg = load_v3_config(args.config)
    log_path = cfg.logs_dir / "step_c.log"

    with StepLogger(log_path) as logger:
        try:
            logger.info(f"=== STEP C: Smoke Run ({args.num_images} images per backbone) ===")
            _, _, val_samples, eval_samples = create_v3_category_splits(
                data_root=cfg.data_root,
                manifest_dir=cfg.manifest_dir,
                seed=42,
            )
            smoke_samples = (val_samples + eval_samples)[: args.num_images]
            if not smoke_samples:
                raise RuntimeError(f"No MVTec samples found under {cfg.data_root}")

            b32_stats = run_backbone_smoke("clip_b32_224", "fp32", smoke_samples, cfg.cache_dir, logger)
            l14_stats = run_backbone_smoke("clip_l14_336", "fp16", smoke_samples, cfg.cache_dir, logger)

            elapsed = time.perf_counter() - t0
            out_json = cfg.results_dir / "step_c_smoke.json"
            out_json.write_text(
                json.dumps({"status": "PASS", "clip_b32_224": b32_stats, "clip_l14_336": l14_stats}, indent=2),
                encoding="utf-8",
            )

            logger.info(f"STEP C RESULT: PASS ({elapsed:.2f}s)")
            log_decision(
                cfg,
                "Step C (Smoke Run 20 imgs/backbone)",
                "PASS",
                elapsed,
                f"B32_VRAM={b32_stats['peak_vram_mb']:.1f}MB, L14_VRAM={l14_stats['peak_vram_mb']:.1f}MB",
            )
            return 0
        except (RuntimeError, OSError, ValueError, KeyError) as exc:
            elapsed = time.perf_counter() - t0
            logger.error(f"Step C failed: {exc}")
            log_decision(cfg, "Step C (Smoke Run 20 imgs/backbone)", "FAIL", elapsed, str(exc))
            return 1


if __name__ == "__main__":
    sys.exit(main())

