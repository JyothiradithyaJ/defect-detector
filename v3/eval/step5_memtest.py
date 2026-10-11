"""Step 5C: 1920x1080 full-pipeline memory ceiling (< 7.0 GB VRAM) and determinism check."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image, ImageDraw

from v3.config import (
    DEFAULT_CONFIG_PATH,
    StepLogger,
    get_git_commit,
    load_v3_config,
    log_decision,
)
from v3.core.bank import DINOFeatureExtractor
from v3.core.contracts import ROI, Recipe
from v3.core.pipeline import TwoStagePipeline

MAX_VRAM_MB = 7.0 * 1024.0  # 7.0 GB ceiling


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 5C: 1920x1080 VRAM (<7GB) and determinism verification.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to v3 YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned 1920x1080 memory and determinism test, then exit.")
    return parser.parse_args()


def make_1080p_test_images() -> tuple[Recipe, list[Image.Image], Image.Image]:
    w, h = 1920, 1080
    recipe = Recipe(
        name="hd_panel_1080p",
        template_path="",
        resolution=(w, h),
        view="fixed",
        symmetry="none",
        stable_roi=[(100, 100), (1820, 100), (1820, 980), (100, 980)],
        rois=[
            ROI(id="stud_1", kind="presence", polygon=[(200, 200), (320, 200), (320, 320), (200, 320)], severity=0, hard=True, params={}),
            ROI(id="panel_surface", kind="surface", polygon=[(400, 200), (1600, 200), (1600, 900), (400, 900)], severity=2, hard=False, params={}),
        ],
    )
    rng = np.random.default_rng(42)
    normals: list[Image.Image] = []
    for _ in range(4):
        arr = rng.integers(110, 130, size=(h, w, 3), dtype=np.uint8)
        im = Image.fromarray(arr, mode="RGB")
        ImageDraw.Draw(im).rectangle([(220, 220), (300, 300)], fill=(220, 200, 80))
        normals.append(im)

    query = normals[0].copy()
    ImageDraw.Draw(query).line([(600, 500), (900, 500)], fill=(200, 200, 200), width=2)
    return recipe, normals, query


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    cfg = load_v3_config(args.config)

    if args.dry_run:
        print(
            f"[DRY-RUN] step5_memtest.py: tests 1920x1080 full pipeline, asserts Peak VRAM < {MAX_VRAM_MB:.0f} MB "
            f"(7.0 GB) and determinism across identical seed/config -> {cfg.results_dir}/step5_memtest.json"
        )
        return 0

    log_path = cfg.logs_dir / "step5_memtest.log"
    with StepLogger(log_path) as logger:
        try:
            logger.info("=== Phase 5 Step C: 1920x1080 Memory (< 7.0 GB VRAM) & Determinism Check ===")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            dino_cfg = cfg.raw.get("dino", {})
            dino_ext = DINOFeatureExtractor(
                backbone_id=str(dino_cfg.get("model", "dinov2_vitl14_reg")),
                short_side=int(dino_cfg.get("short_side", 560)),
                patch_size=int(dino_cfg.get("patch_size", 14)),
                embed_dim=int(dino_cfg.get("embed_dim", 1024)),
                precision=str(cfg.raw.get("precision", "fp16")),
                cache_dir=cfg.dino_cache_dir,
                torch_home=cfg.torch_home,
                lazy_load=False,
            )

            recipe, normals, query = make_1080p_test_images()
            pipe1 = TwoStagePipeline(dino_extractor=dino_ext, scorer_mode="knn_positional")
            pipe1.fit(normals, recipe)
            res1 = pipe1.score(query)

            # Second identical run for determinism verification
            pipe2 = TwoStagePipeline(dino_extractor=dino_ext, scorer_mode="knn_positional")
            pipe2.fit(normals, recipe)
            res2 = pipe2.score(query)

            score_diff = abs(res1.image_score - res2.image_score)
            map_diff = float(np.max(np.abs(res1.anomaly_map - res2.anomaly_map)))
            deterministic = (score_diff == 0.0) and (map_diff == 0.0) and (res1.verdict == res2.verdict)
            if not deterministic:
                raise AssertionError(f"Determinism check failed: score_diff={score_diff}, map_diff={map_diff}")

            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
            dino_ext.unload_model()

            if peak_vram_mb >= MAX_VRAM_MB:
                raise AssertionError(f"Peak VRAM {peak_vram_mb:.1f} MB exceeded 7.0 GB ({MAX_VRAM_MB:.0f} MB) ceiling!")

            payload = {
                "status": "PASS",
                "commit": get_git_commit(),
                "config_hash": cfg.config_hash,
                "resolution": [1920, 1080],
                "peak_vram_mb": round(peak_vram_mb, 1),
                "vram_limit_mb": MAX_VRAM_MB,
                "deterministic": deterministic,
                "score_diff": score_diff,
                "map_max_diff": map_diff,
                "verdict": res1.verdict,
            }
            out_json = cfg.results_dir / "step5_memtest.json"
            out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            elapsed = time.perf_counter() - t0
            logger.info(
                f"1920x1080 Peak VRAM: {peak_vram_mb:.1f} MB (< {MAX_VRAM_MB:.0f} MB) | "
                f"Deterministic: {deterministic} (diff=0.000000)"
            )
            logger.info(f"STEP 5C RESULT: PASS ({elapsed:.2f}s)")
            log_decision(cfg, "Phase 5 Step C (1080p VRAM & Determinism)", "PASS", elapsed, f"VRAM={peak_vram_mb:.1f}MB")
            return 0
        except (RuntimeError, OSError, ValueError, AssertionError) as exc:
            elapsed = time.perf_counter() - t0
            logger.error(f"step5_memtest failed: {exc}")
            log_decision(cfg, "Phase 5 Step C (1080p VRAM & Determinism)", "FAIL", elapsed, str(exc))
            return 1


if __name__ == "__main__":
    sys.exit(main())
