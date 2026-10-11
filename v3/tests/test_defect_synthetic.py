"""Synthetic tests for Stage B DINO patch scoring, aggregation, swap, and thin scratch localization."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from v3.core.bank import CachedDINOFeatures, l2_normalize_np
from v3.core.defect.aggregate import aggregate_patch_map, topk_image_score
from v3.core.defect.knn import GlobalKNNScorer
from v3.core.mismatch.positional_knn import PositionalKNNScorer
from v3.eval.splits import assert_disjoint_references_and_negatives


def test_positional_knn_scores_higher_than_global_on_synthetic_swap() -> None:
    """Verify positional_knn detects a spatial swap that knn_global misses."""
    rng = np.random.default_rng(123)
    grid_h, grid_w, dim = 10, 10, 32

    # Reference feature grid with distinct orthogonal features in top-left vs bottom-right
    base_grid = l2_normalize_np(rng.standard_normal((grid_h, grid_w, dim)).astype(np.float32))
    feat_a = np.zeros(dim, dtype=np.float32)
    feat_a[0] = 1.0
    feat_b = np.zeros(dim, dtype=np.float32)
    feat_b[1] = 1.0

    base_grid[1:3, 1:3, :] = feat_a
    base_grid[7:9, 7:9, :] = feat_b
    ref_feat = CachedDINOFeatures(patch_tokens=base_grid, cls_token=feat_a)

    # Swapped query grid: swap top-left block and bottom-right block
    swapped_grid = base_grid.copy()
    swapped_grid[1:3, 1:3, :] = feat_b
    swapped_grid[7:9, 7:9, :] = feat_a

    global_scorer = GlobalKNNScorer(agg_radius=0, topk_frac=0.05)
    pos_scorer = PositionalKNNScorer(radius=2, agg_radius=0, topk_frac=0.05)

    global_scorer.fit_features([ref_feat])
    pos_scorer.fit_features([ref_feat])

    s_global, _ = global_scorer.score_patch_grid(swapped_grid)
    s_pos, _ = pos_scorer.score_patch_grid(swapped_grid)

    assert s_pos > s_global + 0.5, f"Expected positional score ({s_pos:.4f}) >> global score ({s_global:.4f})"


def test_patch_aggregation_and_topk_score() -> None:
    """Verify neighbourhood aggregation radii [1, 3] and top-k image score."""
    patch_map = np.zeros((12, 12), dtype=np.float32)
    patch_map[6, 6] = 9.0

    r0 = aggregate_patch_map(patch_map, radius=0)
    r1 = aggregate_patch_map(patch_map, radius=1)
    r3 = aggregate_patch_map(patch_map, radius=3)

    assert np.isclose(r0[6, 6], 9.0)
    assert np.isclose(r1[6, 6], 1.0)  # 9.0 / (3x3)
    assert r3[6, 6] < r1[6, 6]
    assert np.isclose(topk_image_score(r0, topk_frac=0.001), 9.0)


def test_fewshot_reference_negative_disjointness() -> None:
    """Verify reference normals removed from test negatives pass assertion and overlap raises."""
    all_normals = [f"normal_{i:03d}" for i in range(10)]
    refs = all_normals[:4]
    test_negatives = [x for x in all_normals if x not in set(refs)]
    assert_disjoint_references_and_negatives(refs, test_negatives)

    with pytest.raises(AssertionError, match="Data leakage detected!"):
        assert_disjoint_references_and_negatives(refs, all_normals)


def test_synthetic_scratch_detection() -> None:
    """Verify thin low-contrast scratch line produces anomaly map peak directly on the scratch."""
    w, h = 160, 160
    normal_img = Image.new("RGB", (w, h), color=(128, 128, 128))
    scratch_img = normal_img.copy()
    draw = ImageDraw.Draw(scratch_img)
    # Draw thin low-contrast line horizontally across row y=80 (from x=40 to x=120)
    draw.line([(40, 80), (120, 80)], fill=(142, 142, 142), width=1)

    scorer = GlobalKNNScorer(agg_radius=1, topk_frac=0.01)
    from v3.core.contracts import Recipe

    dummy_recipe = Recipe(
        name="synthetic_surface",
        template_path="",
        resolution=(w, h),
        view="fixed",
        symmetry="none",
        stable_roi=[(0, 0), (w, 0), (w, h), (0, h)],
        rois=[],
    )
    scorer.fit([normal_img], dummy_recipe)
    res_normal = scorer.score(normal_img)
    res_scratch = scorer.score(scratch_img)

    assert res_scratch.image_score > res_normal.image_score
    peak_y, peak_x = np.unravel_index(int(np.argmax(res_scratch.anomaly_map)), res_scratch.anomaly_map.shape)
    assert abs(peak_y - 80) <= 8, f"Expected anomaly peak near scratch y=80, got y={peak_y}"
    assert 32 <= peak_x <= 128, f"Expected anomaly peak within scratch x=[40..120], got x={peak_x}"
