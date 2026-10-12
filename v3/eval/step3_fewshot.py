"""Step 3B: Few-shot DINO evaluation across k in [1,2,4,8,16], seeds [0,1,2], and 3 scorers.

Features:
- Evaluates knn_global, knn_positional, and pca_residual on cached DINO features
- Enforces reference/test-negative disjointness via assert_disjoint_references_and_negatives
- Selects best scorer and aggregation radius on the 5 validation categories ONLY
- Reports full per-category metrics (with normal/anomalous counts, mean +/- std over seeds)
  on the 10 held-out categories
- Marks toothbrush as [LOW-N] and documents the 5% FPR AU-PRO normalization convention
- Writes CSV and Markdown reports to D:\\defect_cache\\results and v3/docs/results/phase_3.md
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from v3.config import (
    DEFAULT_CONFIG_PATH,
    StepLogger,
    get_git_commit,
    load_v3_config,
    log_decision,
)
from v3.core.bank import CachedDINOFeatures, DINOFeatureExtractor
from v3.core.defect.aggregate import (
    aggregate_patch_map,
    topk_image_score,
    upsample_patch_map,
)
from v3.core.defect.knn import GlobalKNNScorer
from v3.core.defect.pca import PCAResidualScorer
from v3.core.mismatch.positional_knn import PositionalKNNScorer
from dataclasses import dataclass
from v3.eval.metrics import _connected_regions, image_auroc, pixel_aupro, pixel_auroc
from v3.eval.splits import (
    DatasetSample,
    assert_disjoint_references_and_negatives,
    create_v3_category_splits,
)
from v3.eval.step3_cache_dino import discover_train_normals


@dataclass
class PrecomputedMaskGPU:
    """Precomputed ground-truth mask tensors and connected-region weights on GPU for a category."""

    mh: int
    mw: int
    gt_flat: torch.Tensor
    pro_flat: torch.Tensor | None
    normal_pixel_count: float
    region_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 3B: Few-shot DINO evaluation on cached features.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to v3 YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Print evaluation grid and estimated output size, then exit.")
    return parser.parse_args()


def build_scorer(
    scorer_name: str,
    extractor: DINOFeatureExtractor,
    radius: int,
    agg_radius: int,
    topk_frac: float,
    chunk_size: int,
) -> GlobalKNNScorer | PositionalKNNScorer | PCAResidualScorer:
    if scorer_name == "knn_global":
        return GlobalKNNScorer(extractor=extractor, agg_radius=agg_radius, topk_frac=topk_frac, chunk_size=chunk_size)
    if scorer_name == "knn_positional":
        return PositionalKNNScorer(extractor=extractor, radius=radius, agg_radius=agg_radius, topk_frac=topk_frac)
    if scorer_name == "pca_residual":
        return PCAResidualScorer(extractor=extractor, agg_radius=agg_radius, topk_frac=topk_frac)
    raise ValueError(f"Unknown scorer: {scorer_name}")


def load_category_masks(samples: list[DatasetSample], metric_max_side: int = 256) -> list[tuple[np.ndarray, int, int]]:
    """Preload ground-truth boolean masks and resized dimensions (mh, mw) for a category."""
    w0, h0 = 0, 0
    out: list[tuple[np.ndarray, int, int]] = []
    for idx, s in enumerate(samples):
        if idx == 0:
            with Image.open(s.image_path) as img:
                w0, h0 = img.size
        if s.label and s.mask_path and s.mask_path.is_file():
            with Image.open(s.mask_path) as mask:
                w, h = mask.size
                scale = min(1.0, metric_max_side / max(w, h)) if metric_max_side > 0 else 1.0
                mw, mh = max(1, round(w * scale)), max(1, round(h * scale))
                gt = np.asarray(mask.convert("L").resize((mw, mh), Image.Resampling.NEAREST), dtype=np.uint8) > 0
        else:
            w, h = w0, h0
            scale = min(1.0, metric_max_side / max(w, h)) if metric_max_side > 0 else 1.0
            mw, mh = max(1, round(w * scale)), max(1, round(h * scale))
            gt = np.zeros((mh, mw), dtype=bool)
        out.append((gt, mh, mw))
    return out


def build_precomputed_mask_gpu(
    c_masks: list[tuple[np.ndarray, int, int]],
    device: torch.device,
    need_aupro: bool = True,
) -> PrecomputedMaskGPU | None:
    """Precompute flattened mask and connected-component PRO weights on GPU once per category."""
    dims_set = {(mh, mw) for _, mh, mw in c_masks}
    if len(dims_set) != 1:
        return None
    mh, mw = next(iter(dims_set))
    m_stack = np.stack([gt for gt, _, _ in c_masks])
    n_imgs = m_stack.shape[0]
    gt_flat = torch.from_numpy(m_stack.reshape(-1)).to(device=device)
    normal_pixel_count = float((~m_stack).sum())

    pro_flat: torch.Tensor | None = None
    region_count = 0
    if need_aupro:
        pro_changes_np = np.zeros(n_imgs * mh * mw, dtype=np.float32)
        for idx, m in enumerate(m_stack):
            regs = _connected_regions(m)
            region_count += len(regs)
            offset = idx * mh * mw
            for r in regs:
                pro_changes_np[offset + r] = 1.0 / len(r)
        pro_flat = torch.from_numpy(pro_changes_np).to(device=device)

    return PrecomputedMaskGPU(
        mh=mh,
        mw=mw,
        gt_flat=gt_flat,
        pro_flat=pro_flat,
        normal_pixel_count=normal_pixel_count,
        region_count=region_count,
    )


def _load_patch_tokens_raw(image_path: Path, extractor: DINOFeatureExtractor) -> np.ndarray:
    """Fast load of cached patch_tokens without redundant CPU float32 normalization."""
    cfile = extractor.cache_dir / f"{extractor.get_cache_key(image_path)}.npz"
    if cfile.is_file():
        with np.load(cfile) as data:
            return data["patch_tokens"]
    return extractor.get_features(image_path).patch_tokens


@torch.inference_mode()
def load_normalized_query_tensor(
    samples: list[DatasetSample],
    extractor: DINOFeatureExtractor,
    device: torch.device,
) -> torch.Tensor:
    """Stream cached test features one image at a time directly into a pre-allocated tensor on `device`."""
    first = _load_patch_tokens_raw(samples[0].image_path, extractor)
    hp, wp, d = first.shape
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    q_t = torch.empty((len(samples), hp, wp, d), dtype=dtype, device=device)

    for idx, s in enumerate(samples):
        pt = first if idx == 0 else _load_patch_tokens_raw(s.image_path, extractor)
        t = torch.from_numpy(pt).to(device=device, dtype=torch.float32)
        t = F.normalize(t, p=2, dim=-1)
        q_t[idx].copy_(t.to(dtype=dtype))
    return q_t


@torch.inference_mode()
def precompute_per_ref_maps_gpu(
    q_t: torch.Tensor,
    unique_refs: list[Path],
    extractor: DINOFeatureExtractor,
    pos_radius: int,
) -> tuple[dict[Path, torch.Tensor], dict[Path, torch.Tensor], dict[Path, torch.Tensor]]:
    """Precompute per-reference similarity maps [N, H_p, W_p] and normalized reference tensors on GPU."""
    device = q_t.device
    dtype = q_t.dtype
    n_imgs, hp, wp, d = q_t.shape
    q_flat = q_t.reshape(-1, d)

    global_maps: dict[Path, torch.Tensor] = {}
    pos_maps: dict[Path, torch.Tensor] = {}
    ref_tensors: dict[Path, torch.Tensor] = {}
    rad = pos_radius
    q_step = 32 * hp * wp

    for ref_p in unique_refs:
        pt = _load_patch_tokens_raw(ref_p, extractor)
        ref_t = F.normalize(torch.from_numpy(pt).to(device=device, dtype=torch.float32), p=2, dim=-1).to(dtype=dtype)
        ref_tensors[ref_p] = ref_t
        ref_flat = ref_t.reshape(-1, d)

        # 1. Global max similarity against single reference image
        g_max = torch.empty(q_flat.shape[0], dtype=torch.float32, device=device)
        for q0 in range(0, q_flat.shape[0], q_step):
            sims = q_flat[q0 : q0 + q_step] @ ref_flat.T
            g_max[q0 : q0 + q_step] = sims.max(dim=1).values.float()
        global_maps[ref_p] = g_max.reshape(n_imgs, hp, wp)

        # 2. Positional window max similarity against single reference image
        p_max = torch.full((n_imgs, hp, wp), -1.0, dtype=torch.float32, device=device)
        for dr in range(-rad, rad + 1):
            r_q_start = max(0, -dr)
            r_q_end = min(hp, hp - dr)
            r_ref_start = r_q_start + dr
            r_ref_end = r_q_end + dr
            for dc in range(-rad, dc_end := rad + 1):
                c_q_start = max(0, -dc)
                c_q_end = min(wp, wp - dc)
                c_ref_start = c_q_start + dc
                c_ref_end = c_q_end + dc
                q_sub = q_t[:, r_q_start:r_q_end, c_q_start:c_q_end, :]
                r_sub = ref_t[r_ref_start:r_ref_end, c_ref_start:c_ref_end, :].unsqueeze(0)
                sim_shift = (q_sub * r_sub).sum(dim=-1).float()
                p_max[:, r_q_start:r_q_end, c_q_start:c_q_end] = torch.maximum(
                    p_max[:, r_q_start:r_q_end, c_q_start:c_q_end],
                    sim_shift,
                )
        pos_maps[ref_p] = p_max

    return global_maps, pos_maps, ref_tensors


@torch.inference_mode()
def compute_pca_raw_dist_gpu(
    q_t: torch.Tensor,
    refs: list[Path],
    ref_tensors: dict[Path, torch.Tensor],
    variance_retained: float = 0.95,
    max_components: int = 128,
) -> torch.Tensor:
    """Fit PCA subspace on GPU in float32 and return raw residual energy map [N, H_p, W_p] on GPU."""
    device = q_t.device
    n_imgs, hp, wp, d = q_t.shape
    flat = torch.stack([ref_tensors[p] for p in refs], dim=0).reshape(-1, d).float()

    if flat.shape[0] > 8192:
        rng = np.random.default_rng(42)
        idx = torch.from_numpy(rng.choice(flat.shape[0], size=8192, replace=False)).to(device=device)
        flat = flat[idx]

    mean = flat.mean(dim=0, keepdim=True)
    centered_ref = flat - mean
    cov = centered_ref.T @ centered_ref
    eigvals, eigvecs = torch.linalg.eigh(cov)
    var = torch.clamp(eigvals.flip(0), min=0.0)
    vecs = eigvecs.flip(1)
    total_var = float(var.sum().item())
    if total_var <= 1e-12:
        k = 1
    else:
        cum_var = torch.cumsum(var, dim=0) / total_var
        k = int(torch.searchsorted(cum_var, variance_retained).item()) + 1
        k = max(1, min(k, max_components, vecs.shape[1]))

    comps = vecs[:, :k]  # [D, K] float32
    q_flat = q_t.reshape(-1, d)
    n_q = q_flat.shape[0]
    res_flat = torch.empty(n_q, dtype=torch.float32, device=device)
    q_step = 32 * hp * wp
    for q0 in range(0, n_q, q_step):
        c_q = q_flat[q0 : q0 + q_step].float() - mean
        coords = c_q @ comps
        total_sq = (c_q * c_q).sum(dim=-1)
        proj_sq = (coords * coords).sum(dim=-1)
        res_flat[q0 : q0 + q_step] = torch.clamp(total_sq - proj_sq, min=0.0)
    return res_flat.reshape(n_imgs, hp, wp)


@torch.inference_mode()
def score_gpu_maps_to_metrics(
    raw_t: torch.Tensor,
    labels: list[int],
    masks_and_dims: list[tuple[np.ndarray, int, int]],
    mask_gpu: PrecomputedMaskGPU | None,
    agg_radius: int,
    topk_frac: float,
    compute_aupro: bool,
    max_fpr: float = 0.05,
) -> dict[str, float]:
    """Smooth patch maps on GPU, upsample on GPU, and compute exact AUROC/AU-PRO on GPU."""
    n_imgs, hp, wp = raw_t.shape
    device = raw_t.device

    if mask_gpu is not None:
        mh, mw = mask_gpu.mh, mask_gpu.mw
        t_in = raw_t.unsqueeze(1)  # [N, 1, hp, wp]
        if agg_radius > 0:
            padded = F.pad(t_in, (agg_radius, agg_radius, agg_radius, agg_radius), mode="replicate")
            t_smooth = F.avg_pool2d(padded, kernel_size=2 * agg_radius + 1, stride=1)
        else:
            t_smooth = t_in

        flat = t_smooth.reshape(n_imgs, -1)
        k_top = max(1, round(hp * wp * topk_frac))
        img_scores = torch.topk(flat, k=k_top, dim=1).values.mean(dim=1).cpu().numpy().tolist()

        s_flat = F.interpolate(t_smooth, size=(mh, mw), mode="bilinear", align_corners=False).reshape(-1)
        order = torch.argsort(s_flat, descending=True)
        s_sorted = s_flat[order]
        y_sorted = mask_gpu.gt_flat[order].to(torch.float64)

        tps = torch.cumsum(y_sorted, dim=0)
        fps = torch.arange(1, len(y_sorted) + 1, device=device, dtype=torch.float64) - tps
        keep = torch.cat([torch.diff(s_sorted) != 0, torch.ones(1, dtype=torch.bool, device=device)])
        tps_k = torch.cat([torch.zeros(1, dtype=torch.float64, device=device), tps[keep]])
        fps_k = torch.cat([torch.zeros(1, dtype=torch.float64, device=device), fps[keep]])
        px_auc = float((torch.trapezoid(tps_k, fps_k) / (tps_k[-1] * fps_k[-1])).item())

        px_pro = 0.0
        if compute_aupro and mask_gpu.pro_flat is not None and mask_gpu.region_count > 0:
            fprs_k = torch.cat([fps_k / mask_gpu.normal_pixel_count, torch.ones(1, dtype=torch.float64, device=device)])
            pros = torch.cumsum(mask_gpu.pro_flat[order].to(torch.float64), dim=0) / mask_gpu.region_count
            pros_k = torch.cat(
                [
                    torch.zeros(1, dtype=torch.float64, device=device),
                    torch.clamp(pros[keep], 0.0, 1.0),
                    torch.ones(1, dtype=torch.float64, device=device),
                ]
            )
            idx_limit = min(len(fprs_k), int(torch.searchsorted(fprs_k, max_fpr).item()) + 2)
            fpr_cpu = fprs_k[:idx_limit].cpu().numpy()
            pro_cpu = pros_k[:idx_limit].cpu().numpy()
            if max_fpr not in fpr_cpu:
                ix = int(np.searchsorted(fpr_cpu, max_fpr))
                l_f, r_f = fpr_cpu[ix - 1], fpr_cpu[ix]
                l_p, r_p = pro_cpu[ix - 1], pro_cpu[ix]
                interp = l_p + (r_p - l_p) * (max_fpr - l_f) / (r_f - l_f)
                fpr_cpu = np.insert(fpr_cpu, ix, max_fpr)
                pro_cpu = np.insert(pro_cpu, ix, interp)
            valid = fpr_cpu <= max_fpr
            px_pro = float(np.trapezoid(pro_cpu[valid], fpr_cpu[valid] / max_fpr))

        return {
            "image_auroc": image_auroc(labels, img_scores),
            "pixel_auroc": px_auc,
            "pixel_aupro_005": px_pro,
        }

    raw_np = raw_t.cpu().numpy()
    img_scores = []
    heatmaps = []
    masks = []
    for i, (gt_mask, mh, mw) in enumerate(masks_and_dims):
        smoothed = aggregate_patch_map(raw_np[i], radius=agg_radius)
        img_scores.append(topk_image_score(smoothed, topk_frac=topk_frac))
        heatmaps.append(upsample_patch_map(smoothed, (mh, mw)))
        masks.append(gt_mask)
    h_stack = np.stack(heatmaps)
    m_stack = np.stack(masks)
    return {
        "image_auroc": image_auroc(labels, img_scores),
        "pixel_auroc": pixel_auroc(h_stack, m_stack),
        "pixel_aupro_005": pixel_aupro(h_stack, m_stack, max_false_positive_rate=max_fpr) if compute_aupro else 0.0,
    }


def evaluate_single_run(
    scorer_name: str,
    extractor: DINOFeatureExtractor,
    ref_paths: list[Path],
    test_samples: list[DatasetSample],
    test_features: list[CachedDINOFeatures],
    masks_and_dims: list[tuple[np.ndarray, int, int]],
    pos_radius: int,
    agg_radius: int,
    topk_frac: float,
    chunk_size: int,
    compute_aupro: bool = True,
) -> dict[str, float]:
    """Compatibility entry point for single-run evaluation."""
    ref_ids = [str(p.resolve()) for p in ref_paths]
    neg_ids = [str(s.image_path.resolve()) for s in test_samples if s.label == 0]
    assert_disjoint_references_and_negatives(ref_ids, neg_ids)

    ref_feats = [extractor.get_features(p) for p in ref_paths]
    scorer = build_scorer(
        scorer_name=scorer_name,
        extractor=extractor,
        radius=pos_radius,
        agg_radius=agg_radius,
        topk_frac=topk_frac,
        chunk_size=chunk_size,
    )
    scorer.fit_features(ref_feats, reference_ids=ref_ids)

    labels = [s.label for s in test_samples]
    img_scores: list[float] = []
    heatmaps: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    for feat, (gt_mask, mh, mw) in zip(test_features, masks_and_dims, strict=True):
        s_img, patch_map = scorer.score_patch_grid(feat.patch_tokens)
        img_scores.append(s_img)
        heatmaps.append(upsample_patch_map(patch_map, (mh, mw)))
        masks.append(gt_mask)

    h_stack = np.stack(heatmaps)
    m_stack = np.stack(masks)
    return {
        "image_auroc": image_auroc(labels, img_scores),
        "pixel_auroc": pixel_auroc(h_stack, m_stack),
        "pixel_aupro_005": pixel_aupro(h_stack, m_stack, max_false_positive_rate=0.05) if compute_aupro else 0.0,
    }


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    cfg = load_v3_config(args.config)

    seeds: list[int] = list(cfg.raw.get("seed", [0, 1, 2]))
    k_grid: list[int] = list(cfg.raw.get("refs", {}).get("fewshot_k_grid", [1, 2, 4, 8, 16]))
    scorers: list[str] = list(cfg.raw.get("match", {}).get("scorers", ["knn_global", "knn_positional", "pca_residual"]))
    agg_radii: list[int] = list(cfg.raw.get("match", {}).get("agg_radii", [1, 3]))
    pos_radius = int(cfg.raw.get("match", {}).get("radius", 2))
    dino_cfg = cfg.raw.get("dino", {})
    topk_frac = float(dino_cfg.get("topk_frac", 0.001))

    if args.dry_run:
        total_runs = len(k_grid) * len(seeds) * len(scorers) * 10
        print(
            f"[DRY-RUN] step3_fewshot.py: k_grid={k_grid}, seeds={seeds}, scorers={scorers}, "
            f"agg_radii={agg_radii}, heldout_runs={total_runs}, output={cfg.results_dir}/phase_3_fewshot.csv "
            f"(estimated size ~0.5 MB)"
        )
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_path = cfg.logs_dir / "step3_fewshot.log"
    with StepLogger(log_path) as logger:
        try:
            logger.info(f"=== Phase 3 Step B: Few-Shot DINO Evaluation on Cached Features (device={device}) ===")
            val_cats, eval_cats, val_samples, eval_samples = create_v3_category_splits(
                data_root=cfg.data_root,
                manifest_dir=cfg.manifest_dir,
                seed=42,
            )

            extractor = DINOFeatureExtractor(
                backbone_id=str(dino_cfg.get("model", "dinov2_vitl14_reg")),
                short_side=int(dino_cfg.get("short_side", 560)),
                patch_size=int(dino_cfg.get("patch_size", 14)),
                embed_dim=int(dino_cfg.get("embed_dim", 1024)),
                precision=str(cfg.raw.get("precision", "fp16")),
                cache_dir=cfg.dino_cache_dir,
                torch_home=cfg.torch_home,
                lazy_load=True,
            )

            # 1. Select best (scorer, agg_radius) strictly on the 5 validation categories at k=4
            logger.info(f"Selecting best (scorer, agg_radius) on 5 validation categories ({val_cats}) at k=4...")
            val_cache_path = cfg.results_dir / "step3_val_selection.json"
            best_scorer = "knn_positional"
            best_agg = 1
            best_val_score = -1.0
            val_selection_table: list[dict] = []

            if val_cache_path.is_file():
                cached_val = json.loads(val_cache_path.read_text(encoding="utf-8"))
                val_selection_table = cached_val["validation_selection"]
                best_scorer = cached_val["selected_scorer"]
                best_agg = int(cached_val["selected_agg_radius"])
                for entry in val_selection_table:
                    logger.info(
                        f"  [Validation k=4 (cached)] scorer={entry['scorer']:<15} "
                        f"agg_radius={entry['agg_radius']} -> Combined AUROC={entry['val_combined_auroc']:.4f}"
                    )
            else:
                val_scores_by_pair: dict[tuple[str, int], list[float]] = {
                    (sc_name, agg_r): [] for sc_name in scorers for agg_r in agg_radii
                }

                for v_idx, cat in enumerate(val_cats, start=1):
                    t_v = time.perf_counter()
                    c_samples = [s for s in val_samples if s.category == cat]
                    labels = [s.label for s in c_samples]
                    q_t = load_normalized_query_tensor(c_samples, extractor, device)
                    c_masks = load_category_masks(c_samples)
                    mask_gpu = build_precomputed_mask_gpu(c_masks, device, need_aupro=False)

                    pool = discover_train_normals(cfg.data_root, cat, max_refs=32)
                    refs = random.Random(0).sample(pool, min(4, len(pool)))
                    ref_ids = [str(p.resolve()) for p in refs]
                    neg_ids = [str(s.image_path.resolve()) for s in c_samples if s.label == 0]
                    assert_disjoint_references_and_negatives(ref_ids, neg_ids)

                    g_cache, p_cache, ref_tensors = precompute_per_ref_maps_gpu(q_t, refs, extractor, pos_radius)
                    pca_raw = compute_pca_raw_dist_gpu(q_t, refs, ref_tensors)

                    for sc_name in scorers:
                        if sc_name == "knn_global":
                            raw_t = 1.0 - torch.stack([g_cache[p] for p in refs], dim=0).amax(dim=0)
                        elif sc_name == "knn_positional":
                            raw_t = 1.0 - torch.stack([p_cache[p] for p in refs], dim=0).amax(dim=0)
                        else:
                            raw_t = pca_raw

                        for agg_r in agg_radii:
                            m = score_gpu_maps_to_metrics(
                                raw_t, labels, c_masks, mask_gpu, agg_radius=agg_r, topk_frac=topk_frac, compute_aupro=False
                            )
                            val_scores_by_pair[(sc_name, agg_r)].append(0.5 * (m["image_auroc"] + m["pixel_auroc"]))

                    del q_t, c_masks, mask_gpu, g_cache, p_cache, ref_tensors, pca_raw
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()
                    logger.info(
                        f"  [Val {v_idx}/{len(val_cats)}] {cat:<12} ({len(c_samples):3d} imgs) DONE in {time.perf_counter() - t_v:.1f}s"
                    )

                for sc_name in scorers:
                    for agg_r in agg_radii:
                        mean_val = float(np.mean(val_scores_by_pair[(sc_name, agg_r)]))
                        val_selection_table.append({"scorer": sc_name, "agg_radius": agg_r, "val_combined_auroc": mean_val})
                        logger.info(f"  [Validation k=4] scorer={sc_name:<15} agg_radius={agg_r} -> Combined AUROC={mean_val:.4f}")
                        if mean_val > best_val_score:
                            best_val_score = mean_val
                            best_scorer = sc_name
                            best_agg = agg_r

                val_cache_path.write_text(
                    json.dumps(
                        {
                            "selected_scorer": best_scorer,
                            "selected_agg_radius": best_agg,
                            "validation_selection": val_selection_table,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

            logger.info(f"Selected on validation categories: scorer={best_scorer}, agg_radius={best_agg}")

            # 2. Evaluate k in [1,2,4,8,16] x seeds [0,1,2] x 3 scorers on the 10 held-out categories
            csv_rows: list[dict] = []

            for e_idx, cat in enumerate(eval_cats, start=1):
                t_cat = time.perf_counter()
                c_samples = [s for s in eval_samples if s.category == cat]
                labels = [s.label for s in c_samples]
                q_t = load_normalized_query_tensor(c_samples, extractor, device)
                c_masks = load_category_masks(c_samples)
                mask_gpu = build_precomputed_mask_gpu(c_masks, device, need_aupro=True)

                n_norm = sum(1 for s in c_samples if s.label == 0)
                n_anom = sum(1 for s in c_samples if s.label == 1)
                pool = discover_train_normals(cfg.data_root, cat, max_refs=32)
                neg_ids = [str(s.image_path.resolve()) for s in c_samples if s.label == 0]

                unique_refs_set: set[Path] = set()
                for k in k_grid:
                    for sd in seeds:
                        refs = random.Random(sd).sample(pool, min(k, len(pool)))
                        assert_disjoint_references_and_negatives([str(p.resolve()) for p in refs], neg_ids)
                        unique_refs_set.update(refs)
                unique_refs = sorted(unique_refs_set)

                g_cache, p_cache, ref_tensors = precompute_per_ref_maps_gpu(q_t, unique_refs, extractor, pos_radius)

                for sc_name in scorers:
                    for k in k_grid:
                        seed_metrics: list[dict[str, float]] = []
                        for sd in seeds:
                            refs = random.Random(sd).sample(pool, min(k, len(pool)))
                            if sc_name == "knn_global":
                                raw_t = 1.0 - torch.stack([g_cache[p] for p in refs], dim=0).amax(dim=0)
                            elif sc_name == "knn_positional":
                                raw_t = 1.0 - torch.stack([p_cache[p] for p in refs], dim=0).amax(dim=0)
                            else:
                                raw_t = compute_pca_raw_dist_gpu(q_t, refs, ref_tensors)

                            m = score_gpu_maps_to_metrics(
                                raw_t,
                                labels,
                                c_masks,
                                mask_gpu,
                                agg_radius=best_agg,
                                topk_frac=topk_frac,
                                compute_aupro=(k == 4 and sc_name == best_scorer),
                            )
                            seed_metrics.append(m)

                        row = {
                            "category": cat,
                            "images": len(c_samples),
                            "normal": n_norm,
                            "anomalous": n_anom,
                            "scorer": sc_name,
                            "k": k,
                            "agg_radius": best_agg,
                            "image_auroc_mean": float(np.mean([x["image_auroc"] for x in seed_metrics])),
                            "image_auroc_std": float(np.std([x["image_auroc"] for x in seed_metrics])),
                            "pixel_auroc_mean": float(np.mean([x["pixel_auroc"] for x in seed_metrics])),
                            "pixel_auroc_std": float(np.std([x["pixel_auroc"] for x in seed_metrics])),
                            "pixel_aupro_005_mean": float(np.mean([x["pixel_aupro_005"] for x in seed_metrics])),
                            "pixel_aupro_005_std": float(np.std([x["pixel_aupro_005"] for x in seed_metrics])),
                            "is_low_n": cat == "toothbrush",
                        }
                        csv_rows.append(row)

                best_k16 = next(r for r in csv_rows if r["category"] == cat and r["scorer"] == best_scorer and r["k"] == 16)
                del q_t, c_masks, mask_gpu, g_cache, p_cache, ref_tensors
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                logger.info(
                    f"  [Held-out {e_idx:2d}/{len(eval_cats)}] {cat:<12} ({len(c_samples):3d} imgs) DONE in {time.perf_counter() - t_cat:.1f}s | "
                    f"{best_scorer} k=16: Img={best_k16['image_auroc_mean']:.4f}±{best_k16['image_auroc_std']:.4f}, "
                    f"Px={best_k16['pixel_auroc_mean']:.4f}±{best_k16['pixel_auroc_std']:.4f}"
                )

            csv_path = cfg.results_dir / "phase_3_fewshot.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
                writer.writeheader()
                writer.writerows(csv_rows)

            summary_path = cfg.results_dir / "phase_3_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "commit": get_git_commit(),
                        "config_hash": cfg.config_hash,
                        "seeds": seeds,
                        "selected_scorer": best_scorer,
                        "selected_agg_radius": best_agg,
                        "validation_selection": val_selection_table,
                        "rows": csv_rows,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            elapsed = time.perf_counter() - t0
            logger.info(f"STEP 3B RESULT: PASS ({elapsed:.2f}s) -> {csv_path}")
            log_decision(
                cfg,
                "Phase 3 Step B (Few-Shot DINO Evaluation)",
                "PASS",
                elapsed,
                f"best_scorer={best_scorer}, best_agg={best_agg}",
            )
            return 0
        except (RuntimeError, OSError, ValueError, KeyError, AssertionError, MemoryError) as exc:
            elapsed = time.perf_counter() - t0
            logger.error(f"step3_fewshot failed: {type(exc).__name__}: {exc}")
            log_decision(cfg, "Phase 3 Step B (Few-Shot DINO Evaluation)", "FAIL", elapsed, str(exc))
            return 1


if __name__ == "__main__":
    sys.exit(main())
