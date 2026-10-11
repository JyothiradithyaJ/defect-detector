"""Synthetic tests for Stage 1 alignment, diverged ECC misaligned flag, and CLS variant check."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from v3.core.align import Stage1Aligner, apply_rigid_transform
from v3.core.contracts import Recipe


def _make_patterned_board(w: int = 160, h: int = 160) -> Image.Image:
    img = Image.new("RGB", (w, h), color=(50, 50, 50))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(36, 36), (124, 124)], outline=(220, 220, 220), width=3)
    draw.rectangle([(48, 48), (76, 76)], fill=(230, 80, 60))
    draw.ellipse([(88, 84), (116, 112)], fill=(60, 180, 230))
    return img


def test_alignment_shift_rotation_recovery() -> None:
    """Verify alignment recovers from known shift and rotation within tolerance."""
    template = _make_patterned_board()
    recipe = Recipe(
        name="board_v1",
        template_path="",
        resolution=(160, 160),
        view="fixed",
        symmetry="none",
        stable_roi=[(20, 20), (140, 20), (140, 140), (20, 140)],
        rois=[],
    )

    # Apply a known shift (+6 px right, -4 px up) and rotation (+3 deg)
    perturbed = apply_rigid_transform(template, dx=-6.0, dy=4.0, angle_deg=-3.0)
    aligner = Stage1Aligner(max_align_residual=0.25)
    outcome = aligner.align(perturbed, template, recipe)

    assert not outcome.diverged, f"Expected alignment to converge, got residual={outcome.residual}"
    assert outcome.mismatch is None
    assert abs(outcome.angle_deg - 3.0) <= 2.0
    assert abs(outcome.shift_xy[0] - 6.0) <= 3.0
    assert abs(outcome.shift_xy[1] - (-4.0)) <= 3.0


def test_diverged_alignment_raises_hard_misaligned() -> None:
    """Verify a severely corrupted/diverged image raises hard mismatch `misaligned`."""
    template = _make_patterned_board()
    corrupted = Image.new("RGB", (160, 160), color=(250, 250, 250))
    recipe = Recipe(
        name="board_v1",
        template_path="",
        resolution=(160, 160),
        view="fixed",
        symmetry="none",
        stable_roi=[(20, 20), (140, 20), (140, 140), (20, 140)],
        rois=[],
    )

    aligner = Stage1Aligner(max_align_residual=0.15)
    outcome = aligner.align(corrupted, template, recipe)
    assert outcome.diverged
    assert outcome.mismatch is not None
    assert outcome.mismatch.kind == "misaligned"
    assert outcome.mismatch.hard is True


def test_variant_check_detects_wrong_recipe() -> None:
    """Verify DINO CLS cosine similarity against recipe mean CLS raises `wrong_variant`."""
    aligner = Stage1Aligner(variant_min_cosine=0.85)
    cls_map = {
        "recipe_a": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "recipe_b": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    }
    query_cls_b = np.array([0.05, 0.99, 0.0, 0.0], dtype=np.float32)

    assert aligner.check_variant(query_cls_b, "recipe_b", cls_map) is None
    mm = aligner.check_variant(query_cls_b, "recipe_a", cls_map)
    assert mm is not None
    assert mm.kind == "wrong_variant"
    assert mm.hard is True
