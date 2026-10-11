"""Unit and integration tests for Phase 2 multi-backbone CLIP, vocabulary, and polarity."""

from __future__ import annotations

import numpy as np

from v3.core.features import BACKBONE_SPECS
from v3.core.vocabulary import (
    GENERAL_DEFECT_TERMS,
    MVTEC_OBJECT_NAMES,
    PRODUCT_SPECIFIC_VOCABULARY,
    DomainPromptBank,
)
from v3.eval.metrics import pixel_auroc


def test_backbone_specifications() -> None:
    """Verify that both clip_b32_224 and clip_l14_336 are properly registered."""
    assert "clip_b32_224" in BACKBONE_SPECS
    assert "clip_l14_336" in BACKBONE_SPECS

    b32 = BACKBONE_SPECS["clip_b32_224"]
    assert b32["input_size"] == 224
    assert b32["embedding_dim"] == 512
    assert b32["patch_grid"] == 7

    l14 = BACKBONE_SPECS["clip_l14_336"]
    assert l14["input_size"] == 336
    assert l14["embedding_dim"] == 768
    assert l14["patch_grid"] == 24


def test_domain_defect_vocabulary_coverage() -> None:
    """Verify that domain vocabulary covers all 15 MVTec AD categories."""
    assert len(MVTEC_OBJECT_NAMES) == 15
    assert len(PRODUCT_SPECIFIC_VOCABULARY) == 15

    for cat in MVTEC_OBJECT_NAMES:
        assert cat in PRODUCT_SPECIFIC_VOCABULARY
        flaws = PRODUCT_SPECIFIC_VOCABULARY[cat]
        assert len(flaws) >= 5, f"Category {cat} should have at least 5 domain flaw terms"

    assert len(GENERAL_DEFECT_TERMS) >= 10


def test_domain_prompt_bank_generation() -> None:
    """Verify prompt bank expands generic templates and flaws properly without mocks."""
    class DummyExtractor:
        pass

    bank = DomainPromptBank(DummyExtractor())
    normal_prompts, defect_prompts = bank.get_prompts_for_category("bottle")

    assert len(normal_prompts) > 0
    assert len(defect_prompts) > len(normal_prompts)
    assert any("glass" in p for p in defect_prompts)
    assert any("normal bottle" in p for p in normal_prompts)


def test_polarity_inversion_mathematics() -> None:
    """Demonstrate mathematically that sign negation strictly inverts AUROC: AUROC(-S) = 1 - AUROC(S)."""
    np.random.seed(42)
    # Synthetic anomaly maps and ground-truth masks
    heatmaps = np.random.randn(10, 32, 32)
    masks = (np.random.rand(10, 32, 32) > 0.8).astype(bool)

    auroc_native = pixel_auroc(heatmaps, masks)
    auroc_negated = pixel_auroc(-heatmaps, masks)

    # Invert identity holds to float precision
    assert np.isclose(auroc_native + auroc_negated, 1.0, atol=1e-5), (
        f"Native ({auroc_native:.4f}) + Negated ({auroc_negated:.4f}) must equal 1.0"
    )

