"""Dataset splits, manifest loaders, and data leakage guards."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MVTEC_TEXTURES: tuple[str, ...] = ("carpet", "grid", "leather", "tile", "wood")
MVTEC_OBJECTS: tuple[str, ...] = (
    "bottle",
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
)
MVTEC_ALL_CATEGORIES: tuple[str, ...] = tuple(sorted(MVTEC_TEXTURES + MVTEC_OBJECTS))


@dataclass(frozen=True)
class DatasetSample:
    """Standardized dataset record representation."""

    image_id: str
    category: str
    defect_type: str
    label: int
    image_path: Path
    mask_path: Path | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], data_root: Path) -> DatasetSample:
        mask_rel = data.get("mask_path")
        return cls(
            image_id=str(data["image_id"]),
            category=str(data["category"]),
            defect_type=str(data["defect_type"]),
            label=int(data["label"]),
            image_path=data_root / str(data["image_path"]),
            mask_path=data_root / str(mask_rel) if mask_rel else None,
        )


def assert_disjoint_references_and_negatives(
    reference_ids: Sequence[str] | set[str],
    test_negative_ids: Sequence[str] | set[str],
) -> None:
    """Enforce BUILD_PROMPT Rule 3: Reference normals are removed from test negatives.

    Raises AssertionError if any reference normal ID overlaps with test negatives.
    """
    ref_set = set(reference_ids)
    test_set = set(test_negative_ids)
    overlap = ref_set.intersection(test_set)
    if overlap:
        sample = sorted(overlap)[:5]
        raise AssertionError(
            f"Data leakage detected! {len(overlap)} reference images found in test negatives: {sample}"
        )


def load_v1_native_manifests(
    data_root: Path,
    manifest_dir: Path,
) -> tuple[list[DatasetSample], list[DatasetSample]]:
    """Load the original v1 calibration and evaluation manifests."""
    calib_file = manifest_dir / "calibration.json"
    eval_file = manifest_dir / "evaluation.json"

    if not calib_file.is_file():
        raise FileNotFoundError(f"Calibration manifest missing: {calib_file}")
    if not eval_file.is_file():
        raise FileNotFoundError(f"Evaluation manifest missing: {eval_file}")

    with open(calib_file, "r", encoding="utf-8") as f:
        calib_raw = json.load(f)
    with open(eval_file, "r", encoding="utf-8") as f:
        eval_raw = json.load(f)

    calib_samples = [DatasetSample.from_dict(r, data_root) for r in calib_raw]
    eval_samples = [DatasetSample.from_dict(r, data_root) for r in eval_raw]

    # Verify leakage assertion
    calib_negatives = [s.image_id for s in calib_samples if s.label == 0]
    eval_negatives = [s.image_id for s in eval_samples if s.label == 0]
    assert_disjoint_references_and_negatives(calib_negatives, eval_negatives)

    return calib_samples, eval_samples


def create_v3_category_splits(
    data_root: Path,
    manifest_dir: Path,
    seed: int = 42,
    val_texture_count: int = 2,
    val_object_count: int = 3,
) -> tuple[list[str], list[str], list[DatasetSample], list[DatasetSample]]:
    """Create official v3 category split: validation categories vs held-out evaluation categories.

    Uses stratified random selection by seed:
    2 of 5 textures + 3 of 10 objects = 5 validation categories.
    The remaining 10 categories are held-out evaluation categories.
    All samples from the 5 validation categories are used for validation/calibration.
    All samples from the 10 evaluation categories are used for testing.
    """
    rng = random.Random(seed)
    val_textures = rng.sample(sorted(MVTEC_TEXTURES), val_texture_count)
    val_objects = rng.sample(sorted(MVTEC_OBJECTS), val_object_count)
    val_categories = sorted(val_textures + val_objects)
    eval_categories = sorted([c for c in MVTEC_ALL_CATEGORIES if c not in val_categories])

    # Load all evaluation-eligible samples across both v1 manifests
    calib_samples, eval_samples = load_v1_native_manifests(data_root, manifest_dir)
    all_samples = calib_samples + eval_samples

    val_samples = [s for s in all_samples if s.category in val_categories]
    eval_heldout_samples = [s for s in eval_samples if s.category in eval_categories]

    return val_categories, eval_categories, val_samples, eval_heldout_samples
