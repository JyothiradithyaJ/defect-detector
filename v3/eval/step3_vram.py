"""Step 3C: Peak VRAM and latency benchmark per DINO scorer (knn_global, knn_positional, pca_residual)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from v3.config import DEFAULT_CONFIG_PATH, StepLogger, load_v3_config, log_decision
from v3.core.bank import DINOFeatureExtractor
from v3.eval.splits import create_v3_category_splits
from v3.eval.step3_cache_dino import discover_train_normals
from v3.eval.step3_fewshot import build_scorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 3C: Peak VRAM and latency benchmark per DINO scorer.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to v3 YAML config.")
    parser.add_argument("--num-images", type=int, default=20, help="Number of query images per scorer benchmark.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned benchmark and exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    cfg = load_v3_config(args.config)
    dino_cfg = cfg.raw.get("dino", {})
    scorers = list(cfg.raw.get("match", {}).get("scorers", ["knn_global", "knn_positional", "pca_residual"]))

    if args.dry_run:
        print(
            f"[DRY-RUN] step3_vram.py: scorers={scorers}, num_images={args.num_images}, "
            f"output={cfg.results_dir}/step3_vram.json (estimated size ~4 KB)"
        )
        return 0

    log_path = cfg.logs_dir / "step3_vram.log"
    with StepLogger(log_path) as logger:
        try:
            logger.info("=== Phase 3 Step C: Peak VRAM & Latency per DINO Scorer ===")
            _, eval_cats, _, eval_samples = create_v3_category_splits(
                data_root=cfg.data_root,
                manifest_dir=cfg.manifest_dir,
                seed=42,
            )
            cat = eval_cats[0]
            ref_paths = discover_train_normals(cfg.data_root, cat, max_refs=4)
            query_samples = [s for s in eval_samples if s.category == cat][: args.num_images]

            extractor = DINOFeatureExtractor(
                backbone_id=str(dino_cfg.get("model", "dinov2_vitl14_reg")),
                short_side=int(dino_cfg.get("short_side", 560)),
                patch_size=int(dino_cfg.get("patch_size", 14)),
                embed_dim=int(dino_cfg.get("embed_dim", 1024)),
                precision=str(cfg.raw.get("precision", "fp16")),
                cache_dir=cfg.dino_cache_dir,
                torch_home=cfg.torch_home,
                lazy_load=False,
            )

            ref_feats = [extractor.get_features(p) for p in ref_paths]
            query_feats = [extractor.get_features(s.image_path) for s in query_samples]

            stats: dict[str, dict] = {}
            for sc_name in scorers:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                sc = build_scorer(
                    scorer_name=sc_name,
                    extractor=extractor,
                    radius=int(cfg.raw.get("match", {}).get("radius", 2)),
                    agg_radius=1,
                    topk_frac=float(dino_cfg.get("topk_frac", 0.001)),
                    chunk_size=int(dino_cfg.get("chunk_size", 4096)),
                )
                t_sc = time.perf_counter()
                sc.fit_features(ref_feats)
                for qf in query_feats:
                    sc.score_patch_grid(qf.patch_tokens)
                sc_elapsed = time.perf_counter() - t_sc
                vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
                sec_img = sc_elapsed / max(1, len(query_feats))
                stats[sc_name] = {
                    "seconds_per_image": round(sec_img, 5),
                    "peak_vram_mb": round(vram_mb, 1),
                }
                logger.info(f"  Scorer [{sc_name:<15}]: {sec_img:.5f} s/img | Peak VRAM: {vram_mb:.1f} MB")

            extractor.unload_model()
            out_json = cfg.results_dir / "step3_vram.json"
            out_json.write_text(json.dumps({"status": "PASS", "scorers": stats}, indent=2), encoding="utf-8")

            elapsed = time.perf_counter() - t0
            logger.info(f"STEP 3C RESULT: PASS ({elapsed:.2f}s)")
            log_decision(cfg, "Phase 3 Step C (DINO Scorer VRAM & Timing)", "PASS", elapsed)
            return 0
        except (RuntimeError, OSError, ValueError) as exc:
            elapsed = time.perf_counter() - t0
            logger.error(f"step3_vram failed: {exc}")
            log_decision(cfg, "Phase 3 Step C (DINO Scorer VRAM & Timing)", "FAIL", elapsed, str(exc))
            return 1


if __name__ == "__main__":
    sys.exit(main())
