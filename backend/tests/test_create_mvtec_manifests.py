"""Unit tests for deterministic MVTec manifest generation."""

from pathlib import Path
import sys

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from scripts import create_mvtec_manifests as manifests  # noqa: E402


def _create_empty_file(path: Path) -> None:
    """Create a placeholder image or mask file for manifest tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


@pytest.fixture
def mini_mvtec_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Create a tiny bottle-only MVTec-like dataset."""
    monkeypatch.setattr(manifests, "MVTEC_CATEGORIES", ("bottle",))

    data_root = tmp_path / "mvtec_ad"

    # Official normal training-reference images.
    _create_empty_file(data_root / "bottle" / "train" / "good" / "000.png")
    _create_empty_file(data_root / "bottle" / "train" / "good" / "001.png")

    # Normal test images.
    _create_empty_file(data_root / "bottle" / "test" / "good" / "000.png")
    _create_empty_file(data_root / "bottle" / "test" / "good" / "001.png")

    # Defective test images and their corresponding masks.
    for image_name in ("000.png", "001.png"):
        _create_empty_file(
            data_root / "bottle" / "test" / "broken_large" / image_name
        )
        _create_empty_file(
            data_root
            / "bottle"
            / "ground_truth"
            / "broken_large"
            / f"{Path(image_name).stem}_mask.png"
        )

    return data_root


def test_manifest_generation_separates_train_reference_split(
    mini_mvtec_root: Path,
) -> None:
    """Train-reference metadata contains only normal training records."""
    records = manifests.build_train_reference_manifest(mini_mvtec_root)

    assert len(records) == 2
    assert all(record["label"] == 0 for record in records)
    assert all(record["defect_type"] == "good" for record in records)
    assert all(record["mask_path"] is None for record in records)


def test_test_split_has_no_calibration_evaluation_overlap(
    mini_mvtec_root: Path,
) -> None:
    """An image must belong to calibration or evaluation, never both."""
    calibration, evaluation = manifests.build_test_manifests(
        data_root=mini_mvtec_root,
        calibration_fraction=0.5,
        seed=42,
    )

    calibration_ids = {record["image_id"] for record in calibration}
    evaluation_ids = {record["image_id"] for record in evaluation}

    assert len(calibration) == 2
    assert len(evaluation) == 2
    assert calibration_ids.isdisjoint(evaluation_ids)

    assert {record["label"] for record in calibration} == {0, 1}
    assert {record["label"] for record in evaluation} == {0, 1}


def test_defective_records_include_ground_truth_masks(
    mini_mvtec_root: Path,
) -> None:
    """Defective MVTec records must reference their matching pixel masks."""
    calibration, evaluation = manifests.build_test_manifests(
        data_root=mini_mvtec_root,
        calibration_fraction=0.5,
        seed=42,
    )

    records = calibration + evaluation
    defective_records = [
        record for record in records if record["label"] == 1
    ]

    assert len(defective_records) == 2
    assert all(record["mask_path"] is not None for record in defective_records)
    assert all(
        str(record["mask_path"]).endswith("_mask.png")
        for record in defective_records
    )


def test_same_seed_produces_the_same_split(
    mini_mvtec_root: Path,
) -> None:
    """The fixed seed makes future experiments reproducible."""
    first_calibration, first_evaluation = manifests.build_test_manifests(
        data_root=mini_mvtec_root,
        calibration_fraction=0.5,
        seed=42,
    )
    second_calibration, second_evaluation = manifests.build_test_manifests(
        data_root=mini_mvtec_root,
        calibration_fraction=0.5,
        seed=42,
    )

    assert first_calibration == second_calibration
    assert first_evaluation == second_evaluation
