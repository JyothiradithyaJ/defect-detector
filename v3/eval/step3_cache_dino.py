"""Step 3A: DINOv2 feature cache builder -> D:\\defect_cache\\dino_features.

Features:
- Caches train normal references and test samples as fp16 .npz files
- Cache key = sha256(backbone:short_side:precision:image_sha256)
- Skip-if-exists, resumable per category, progress line every 30 s
- Supports --dry-run to estimate output disk size before writing anything
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from v3.config import DEFAULT_CONFIG_PATH, StepLogger, load_v3_config, log_decision
from v3.core.bank import DINOFeatureExtractor, estimate_dino_cache_bytes
from v3.eval.splits import MVTEC_CATEGORIES, DatasetSample, discover_mvtec_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 3A: Build DINOv2 fp16 feature cache on D:\\defect_cache\\dino_features.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to v3 YAML config.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size parameter for extraction.")
    parser.add_argument("--max-train-refs", type=int, default=32, help="Number of train/good normals per category to cache.")
    parser.add_argument("--dry-run", action="store_true", help="Estimate output cache size and exit without running.")
    return parser.parse_args()


def discover_train_normals(data_root: Path, category: str, max_refs: int) -> list[Path]:
    """Return sorted train/good image paths for `category` (up to `max_refs`)."""
    good_dir = data_root / category / "train" / "good"
    if not good_dir.is_dir():
        return []
    paths = sorted(p for p in good_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    return paths[:max_refs]


def is_valid_dino_npz(cfile: Path) -> bool:
    if not cfile.is_file() or cfile.stat().st_size < 128:
        return False
    try:
        with zipfile.ZipFile(cfile, "r") as zf:
            return "patch_tokens.npy" in zf.namelist() and "cls_token.npy" in zf.namelist()
    except (OSError, zipfile.BadZipFile):
        cfile.unlink(missing_ok=True)
        return False


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    cfg = load_v3_config(args.config)
    dino_cfg = cfg.raw.get("dino", {})
    backbone = str(dino_cfg.get("model", "dinov2_vitl14_reg"))
    short_side = int(dino_cfg.get("short_side", 560))
    patch_size = int(dino_cfg.get("patch_size", 14))
    embed_dim = int(dino_cfg.get("embed_dim", 1024))
    precision = str(cfg.raw.get("precision", "fp16"))

    # Collect target image count (15 categories: up to max_train_refs train normals + all test samples)
    test_samples: list[DatasetSample] = discover_mvtec_samples(cfg.data_root, MVTEC_CATEGORIES)
    train_paths: dict[str, list[Path]] = {
        cat: discover_train_normals(cfg.data_root, cat, args.max_train_refs)
        for cat in MVTEC_CATEGORIES
    }
    total_images = len(test_samples) + sum(len(v) for v in train_paths.values())
    est_bytes = estimate_dino_cache_bytes(
        num_images=total_images,
        short_side=short_side,
        patch_size=patch_size,
        embed_dim=embed_dim,
    )
    est_gb = est_bytes / (1024**3)

    if args.dry_run:
        grid = short_side // patch_size
        print(
            f"[DRY-RUN] step3_cache_dino.py: backbone={backbone}, short_side={short_side} "
            f"(grid={grid}x{grid}), precision={precision}, total_images={total_images} "
            f"({len(test_samples)} test + {total_images - len(test_samples)} train refs) -> "
            f"target={cfg.dino_cache_dir}, estimated_size={est_gb:.3f} GB"
        )
        return 0

    log_path = cfg.logs_dir / "step3_cache_dino.log"
    with StepLogger(log_path) as logger:
        try:
            logger.info(
                f"=== Phase 3 Step A: DINO Feature Cache Builder ({backbone}, {short_side}px, {precision}) ==="
            )
            logger.info(f"Estimated full cache size for {total_images} images: {est_gb:.3f} GB -> {cfg.dino_cache_dir}")

            extractor = DINOFeatureExtractor(
                backbone_id=backbone,
                short_side=short_side,
                patch_size=patch_size,
                embed_dim=embed_dim,
                precision=precision,
                device="cuda" if torch.cuda.is_available() else "cpu",
                cache_dir=cfg.dino_cache_dir,
                torch_home=cfg.torch_home,
                lazy_load=True,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            skipped = 0
            computed = 0
            last_log_t = time.perf_counter()

            for cat in MVTEC_CATEGORIES:
                cat_test = [s.image_path for s in test_samples if s.category == cat]
                cat_paths = train_paths[cat] + cat_test
                cat_uncached: list[Path] = []
                for p in cat_paths:
                    cfile = cfg.dino_cache_dir / f"{extractor.get_cache_key(p)}.npz"
                    if is_valid_dino_npz(cfile):
                        skipped += 1
                    else:
                        cat_uncached.append(p)

                if not cat_uncached:
                    logger.info(f"  Category [{cat:<12}]: {len(cat_paths)}/{len(cat_paths)} cached (SKIP)")
                    continue

                logger.info(f"  Category [{cat:<12}]: extracting {len(cat_uncached)}/{len(cat_paths)} images...")
                for p in cat_uncached:
                    extractor.get_features(p)
                    computed += 1
                    now = time.perf_counter()
                    if (now - last_log_t) >= 30.0:
                        vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
                        logger.info(
                            f"  [PROGRESS] {skipped + computed}/{total_images} ready "
                            f"(computed={computed}, skipped={skipped}, elapsed={now - t0:.1f}s, "
                            f"peak_vram={vram_mb:.1f} MB)"
                        )
                        last_log_t = now

                vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
                logger.info(f"  Category [{cat:<12}]: DONE (peak_vram={vram_mb:.1f} MB)")

            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
            extractor.unload_model()

            elapsed = time.perf_counter() - t0
            sec_per_img = elapsed / max(1, computed if computed > 0 else total_images)
            out_json = cfg.results_dir / "step3_cache_stats.json"
            out_json.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "backbone": backbone,
                        "short_side": short_side,
                        "precision": precision,
                        "total_images": total_images,
                        "already_cached": skipped,
                        "newly_computed": computed,
                        "elapsed_seconds": round(elapsed, 2),
                        "seconds_per_image": round(sec_per_img, 4),
                        "peak_vram_mb": round(peak_vram_mb, 1),
                        "estimated_size_gb": round(est_gb, 3),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            logger.info(
                f"STEP 3A RESULT: PASS ({elapsed:.2f}s, {sec_per_img:.4f}s/img, Peak VRAM={peak_vram_mb:.1f} MB)"
            )
            log_decision(
                cfg,
                "Phase 3 Step A (DINO Feature Cache)",
                "PASS",
                elapsed,
                f"images={total_images}, peak_vram={peak_vram_mb:.1f}MB",
            )
            return 0
        except (RuntimeError, OSError, ValueError, KeyError) as exc:
            elapsed = time.perf_counter() - t0
            logger.error(f"step3_cache_dino failed: {exc}")
            log_decision(cfg, "Phase 3 Step A (DINO Feature Cache)", "FAIL", elapsed, str(exc))
            return 1


if __name__ == "__main__":
    sys.exit(main())
