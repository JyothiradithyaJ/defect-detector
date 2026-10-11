"""Step D: Embedding cache builder for clip_l14_336 -> D:/defect_cache/visual_embeddings.

Features:
- Stores visual embeddings as fp16 (.npz) in D:/defect_cache/visual_embeddings
- Batch size 8 with automatic fallback to 4 on CUDA OutOfMemoryError
- Skip-if-exists per image and resumable per category
- Emits a progress line at least every 30 seconds and after each category
- Logs peak VRAM, time per image, and total cache size on disk to D:/defect_cache/logs/step_d.log
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
if str(PROJECT_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import numpy as np
import torch
from PIL import Image
from torch.nn import functional

from v3.config import DEFAULT_CONFIG_PATH, StepLogger, load_v3_config, log_decision
from v3.core.features import CLIPFeatureExtractor
from v3.eval.splits import DatasetSample, create_v3_category_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step D: Embedding cache builder for clip_l14_336 (fp16).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to v3 YAML config.")
    parser.add_argument("--backbone", default="clip_l14_336", choices=["clip_l14_336", "clip_b32_224"])
    parser.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--batch-size", type=int, default=8, help="Primary batch size (auto-falls back to 4 on OOM).")
    return parser.parse_args()


def is_valid_cache_file(cfile: Path) -> bool:
    """Quickly verify that an existing .npz file is non-empty and has a valid zip header."""
    if not cfile.is_file() or cfile.stat().st_size < 128:
        return False
    try:
        with zipfile.ZipFile(cfile, "r") as zf:
            return "global_embedding.npy" in zf.namelist()
    except (OSError, zipfile.BadZipFile):
        try:
            cfile.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def extract_batch(
    extractor: CLIPFeatureExtractor,
    chunk: list[tuple[DatasetSample, Path]],
    save_dtype: type,
) -> None:
    """Run forward_intermediates on a batch of images and write .npz files."""
    grid = extractor.spec["patch_grid"]
    dim = extractor.spec["embedding_dim"]
    projection = extractor.model.visual.proj

    orig_tensors = []
    flip_tensors = []
    for sample, _ in chunk:
        with Image.open(sample.image_path) as img:
            orig = img.convert("RGB")
            flip = orig.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            orig_tensors.append(extractor.preprocess(orig))
            flip_tensors.append(extractor.preprocess(flip))

    b_orig = torch.stack(orig_tensors).to(extractor.device)
    b_flip = torch.stack(flip_tensors).to(extractor.device)
    if extractor.precision == "fp16" and extractor.device.type == "cuda":
        b_orig = b_orig.half()
        b_flip = b_flip.half()

    with torch.inference_mode():
        out_orig = extractor.model.forward_intermediates(
            image=b_orig,
            image_indices=3,
            normalize=True,
            normalize_intermediates=True,
            image_output_fmt="NLC",
        )
        out_flip = extractor.model.forward_intermediates(
            image=b_flip,
            image_indices=3,
            normalize=True,
            normalize_intermediates=True,
            image_output_fmt="NLC",
        )

        g_orig = out_orig["image_features"]
        g_flip = out_flip["image_features"]
        inter_orig = out_orig["image_intermediates"][-1]
        inter_flip = out_flip["image_intermediates"][-1]

        p_orig = functional.normalize(inter_orig @ projection, dim=-1)
        p_flip = functional.normalize(inter_flip @ projection, dim=-1)

        if hasattr(extractor.model.visual, "ln_post"):
            f_orig = functional.normalize(extractor.model.visual.ln_post(inter_orig) @ projection, dim=-1)
            f_flip = functional.normalize(extractor.model.visual.ln_post(inter_flip) @ projection, dim=-1)
        else:
            f_orig = p_orig
            f_flip = p_flip

        g_orig_np = g_orig.float().cpu().numpy().astype(save_dtype)
        g_flip_np = g_flip.float().cpu().numpy().astype(save_dtype)
        p_orig_np = p_orig.reshape(-1, grid, grid, dim).float().cpu().numpy().astype(save_dtype)
        p_flip_np = p_flip.reshape(-1, grid, grid, dim).float().cpu().numpy().astype(save_dtype)
        f_orig_np = f_orig.reshape(-1, grid, grid, dim).float().cpu().numpy().astype(save_dtype)
        f_flip_np = f_flip.reshape(-1, grid, grid, dim).float().cpu().numpy().astype(save_dtype)

    for idx, (_, cfile) in enumerate(chunk):
        np.savez(
            cfile,
            global_embedding=g_orig_np[idx],
            global_embedding_flipped=g_flip_np[idx],
            patch_embeddings=p_orig_np[idx],
            patch_embeddings_flipped=p_flip_np[idx],
            final_patch_tokens=f_orig_np[idx],
            final_patch_tokens_flipped=f_flip_np[idx],
        )


def build_cache_for_backbone(
    extractor: CLIPFeatureExtractor,
    all_samples: list[DatasetSample],
    batch_size: int,
    logger: StepLogger,
) -> dict:
    """Resumable per-category batch extraction with OOM fallback and 30s progress logging."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    save_dtype = np.float16 if extractor.precision == "fp16" else np.float32
    categories = sorted({s.category for s in all_samples})
    current_bs = batch_size

    total_images = len(all_samples)
    total_skipped = 0
    total_computed = 0
    t_start = time.perf_counter()
    last_log_t = t_start

    logger.info(
        f"[{extractor.backbone_id} ({extractor.precision})] Starting resumable cache build across "
        f"{len(categories)} categories ({total_images} images) -> {extractor.cache_dir}"
    )

    for cat in categories:
        cat_samples = [s for s in all_samples if s.category == cat]
        uncached: list[tuple[DatasetSample, Path]] = []
        for s in cat_samples:
            cfile = extractor.cache_dir / f"{extractor.get_cache_key(s.image_path)}.npz"
            if is_valid_cache_file(cfile):
                total_skipped += 1
            else:
                uncached.append((s, cfile))

        if not uncached:
            logger.info(f"  Category [{cat:<12}]: {len(cat_samples)}/{len(cat_samples)} already cached (SKIP)")
            continue

        logger.info(
            f"  Category [{cat:<12}]: computing {len(uncached)}/{len(cat_samples)} uncached images "
            f"(batch_size={current_bs})..."
        )

        idx = 0
        while idx < len(uncached):
            chunk = uncached[idx : idx + current_bs]
            try:
                extract_batch(extractor, chunk, save_dtype)
                idx += len(chunk)
                total_computed += len(chunk)
            except torch.OutOfMemoryError:
                if current_bs > 4:
                    current_bs = 4
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.info("  [WARN] CUDA OOM at batch_size=8; auto-falling back to batch_size=4.")
                    continue
                raise

            now = time.perf_counter()
            if (now - last_log_t) >= 30.0:
                vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
                done = total_skipped + total_computed
                logger.info(
                    f"  [PROGRESS] {done}/{total_images} images ready "
                    f"(computed={total_computed}, skipped={total_skipped}, "
                    f"elapsed={now - t_start:.1f}s, peak_vram={vram_mb:.1f} MB)"
                )
                last_log_t = now

        vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
        logger.info(
            f"  Category [{cat:<12}]: DONE ({len(cat_samples)}/{len(cat_samples)} cached, "
            f"peak_vram={vram_mb:.1f} MB)"
        )
        last_log_t = time.perf_counter()

    elapsed = time.perf_counter() - t_start
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    sec_per_img = (elapsed / total_computed) if total_computed > 0 else (elapsed / max(1, total_images))

    # Measure total disk usage for this backbone
    disk_bytes = 0
    for s in all_samples:
        cfile = extractor.cache_dir / f"{extractor.get_cache_key(s.image_path)}.npz"
        if cfile.is_file():
            disk_bytes += cfile.stat().st_size
    disk_gb = disk_bytes / (1024**3)
    disk_mb = disk_bytes / (1024**2)

    return {
        "backbone_id": extractor.backbone_id,
        "precision": extractor.precision,
        "total_images": total_images,
        "already_cached": total_skipped,
        "newly_computed": total_computed,
        "batch_size_used": current_bs,
        "elapsed_seconds": round(elapsed, 2),
        "seconds_per_image": round(sec_per_img, 4),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "disk_size_mb": round(disk_mb, 2),
        "disk_size_gb": round(disk_gb, 3),
    }


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    cfg = load_v3_config(args.config)
    log_path = cfg.logs_dir / "step_d.log"

    with StepLogger(log_path) as logger:
        try:
            logger.info(f"=== STEP D: Embedding Cache Builder ({args.backbone}, {args.precision}) ===")
            _, _, val_samples, eval_samples = create_v3_category_splits(
                data_root=cfg.data_root,
                manifest_dir=cfg.manifest_dir,
                seed=42,
            )
            all_samples = val_samples + eval_samples

            extractor = CLIPFeatureExtractor(
                backbone_id=args.backbone,
                device="cuda" if torch.cuda.is_available() else "cpu",
                precision=args.precision,
                cache_dir=cfg.cache_dir,
            )
            stats = build_cache_for_backbone(extractor, all_samples, args.batch_size, logger)

            # Also verify/ensure clip_b32_224 (fp32) cache is complete so Step E is 100% cache-hit
            del extractor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            ext_b32 = CLIPFeatureExtractor(
                backbone_id="clip_b32_224",
                device="cuda" if torch.cuda.is_available() else "cpu",
                precision="fp32",
                cache_dir=cfg.cache_dir,
            )
            stats_b32 = build_cache_for_backbone(ext_b32, all_samples, args.batch_size, logger)

            out_json = cfg.results_dir / "step_d_cache_stats.json"
            out_json.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "clip_l14_336": stats,
                        "clip_b32_224": stats_b32,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            elapsed = time.perf_counter() - t0
            logger.info(
                f"\nSTEP D RESULT: PASS ({elapsed:.2f}s) | "
                f"L14: {stats['disk_size_gb']:.3f} GB ({stats['seconds_per_image']:.4f}s/img, "
                f"Peak VRAM {stats['peak_vram_mb']:.1f} MB) | "
                f"B32: {stats_b32['disk_size_gb']:.3f} GB"
            )
            log_decision(
                cfg,
                "Step D (L14 fp16 Embedding Cache Build)",
                "PASS",
                elapsed,
                f"L14_size={stats['disk_size_gb']:.3f}GB, L14_VRAM={stats['peak_vram_mb']:.1f}MB, "
                f"sec_per_img={stats['seconds_per_image']:.4f}s",
            )
            return 0
        except (RuntimeError, OSError, ValueError, KeyError) as exc:
            elapsed = time.perf_counter() - t0
            logger.error(f"Step D failed: {exc}")
            log_decision(cfg, "Step D (L14 fp16 Embedding Cache Build)", "FAIL", elapsed, str(exc))
            return 1


if __name__ == "__main__":
    sys.exit(main())

