

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "mvtec_ad"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "manifests"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}

MVTEC_CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)


def parse_arguments() -> argparse.Namespace:
  
    parser = argparse.ArgumentParser(
        description="Create deterministic MVTec AD data manifests."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="MVTec AD dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where JSON manifests are written.",
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.20,
        help="Fraction of each test stratum reserved for calibration.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Fixed seed used for deterministic calibration selection.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing manifest files.",
    )
    return parser.parse_args()


def relative_path(path: Path, data_root: Path) -> str:
    
    return path.relative_to(data_root).as_posix()


def make_record(
    image_path: Path,
    data_root: Path,
    category: str,
    defect_type: str,
    label: int,
    mask_path: Path | None,
) -> dict[str, object]:
    """Create one serialisable image record."""
    image_relative_path = relative_path(image_path, data_root)

    return {
        "image_id": image_relative_path,
        "image_path": image_relative_path,
        "category": category,
        "defect_type": defect_type,
        "label": label,
        "mask_path": (
            relative_path(mask_path, data_root)
            if mask_path is not None
            else None
        ),
    }


def stable_sort_key(image_path: Path, seed: int) -> str:
 
    value = f"{seed}:{image_path.as_posix()}"
    return sha256(value.encode("utf-8")).hexdigest()


def calibration_count(total: int, fraction: float) -> int:
   
    if total < 2:
        return 0

    requested = round(total * fraction)
    return min(max(requested, 1), total - 1)


def test_records_for_stratum(
    data_root: Path,
    category: str,
    defect_type: str,
) -> list[dict[str, object]]:
    """Create records for one test category/defect-type folder."""
    image_directory = data_root / category / "test" / defect_type

    image_paths = sorted(
        path
        for path in image_directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )

    if defect_type == "good":
        return [
            make_record(
                image_path=image_path,
                data_root=data_root,
                category=category,
                defect_type="good",
                label=0,
                mask_path=None,
            )
            for image_path in image_paths
        ]

    records = []

    for image_path in image_paths:
        mask_path = (
            data_root
            / category
            / "ground_truth"
            / defect_type
            / f"{image_path.stem}_mask.png"
        )

        if not mask_path.is_file():
            raise FileNotFoundError(
                f"Ground-truth mask not found for {image_path}: {mask_path}"
            )

        records.append(
            make_record(
                image_path=image_path,
                data_root=data_root,
                category=category,
                defect_type=defect_type,
                label=1,
                mask_path=mask_path,
            )
        )

    return records


def build_train_reference_manifest(data_root: Path) -> list[dict[str, object]]:
    """Legacy helper retained for historical reference-memory experiments."""
    records: list[dict[str, object]] = []

    for category in MVTEC_CATEGORIES:
        good_directory = data_root / category / "train" / "good"

        if not good_directory.is_dir():
            raise FileNotFoundError(
                f"Expected MVTec training directory not found: {good_directory}"
            )

        for image_path in sorted(good_directory.iterdir()):
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            records.append(
                make_record(
                    image_path=image_path,
                    data_root=data_root,
                    category=category,
                    defect_type="good",
                    label=0,
                    mask_path=None,
                )
            )

    return records


def build_test_manifests(
    data_root: Path,
    calibration_fraction: float,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Split every test category/defect-type stratum deterministically."""
    calibration_records: list[dict[str, object]] = []
    evaluation_records: list[dict[str, object]] = []

    for category in MVTEC_CATEGORIES:
        test_directory = data_root / category / "test"

        if not test_directory.is_dir():
            raise FileNotFoundError(
                f"Expected MVTec test directory not found: {test_directory}"
            )

        for defect_directory in sorted(test_directory.iterdir()):
            if not defect_directory.is_dir():
                continue

            defect_type = defect_directory.name
            records = test_records_for_stratum(
                data_root=data_root,
                category=category,
                defect_type=defect_type,
            )

            sorted_records = sorted(
                records,
                key=lambda record: stable_sort_key(
                    Path(str(record["image_path"])),
                    seed,
                ),
            )

            split_index = calibration_count(
                total=len(sorted_records),
                fraction=calibration_fraction,
            )

            calibration_records.extend(sorted_records[:split_index])
            evaluation_records.extend(sorted_records[split_index:])

    return calibration_records, evaluation_records


def assert_no_overlap(
    calibration_records: list[dict[str, object]],
    evaluation_records: list[dict[str, object]],
) -> None:
    """Fail if an image appears in both test manifests."""
    calibration_ids = {
        str(record["image_id"])
        for record in calibration_records
    }
    evaluation_ids = {
        str(record["image_id"])
        for record in evaluation_records
    }

    overlap = calibration_ids & evaluation_ids

    if overlap:
        example = sorted(overlap)[0]
        raise RuntimeError(
            f"Calibration and evaluation manifests overlap: {example}"
        )


def write_json(
    path: Path,
    payload: object,
    overwrite: bool,
) -> None:
    """Write formatted JSON without silently replacing existing manifests."""
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. Use --overwrite to replace it."
        )

    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def manifest_digest(records: list[dict[str, object]]) -> str:
    """Return a reproducibility hash for one manifest."""
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def count_by_label(records: list[dict[str, object]]) -> dict[str, int]:
    """Return readable normal/anomalous image counts."""
    counts = Counter(int(record["label"]) for record in records)

    return {
        "good_label_0": counts[0],
        "defective_label_1": counts[1],
        "total": len(records),
    }


def main() -> None:
    """Generate and save all Phase 1 manifests."""
    args = parse_arguments()

    if not 0.0 < args.calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1.")

    if not args.data_root.is_dir():
        raise FileNotFoundError(
            f"MVTec data root not found: {args.data_root}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    calibration_records, evaluation_records = build_test_manifests(
        data_root=args.data_root,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
    )

    assert_no_overlap(calibration_records, evaluation_records)

    calibration_path = args.output_dir / "calibration.json"
    evaluation_path = args.output_dir / "evaluation.json"
    metadata_path = args.output_dir / "split_metadata.json"

    write_json(calibration_path, calibration_records, args.overwrite)
    write_json(evaluation_path, evaluation_records, args.overwrite)

    metadata = {
        "dataset": "MVTec AD",
        "data_root": str(args.data_root.resolve()),
        "categories": list(MVTEC_CATEGORIES),
        "seed": args.seed,
        "calibration_fraction": args.calibration_fraction,
        "split_policy": (
            "Official test "
            "images are split independently within each category and defect "
            "type using a deterministic SHA-256 ordering."
        ),
        "calibration": {
            "path": calibration_path.name,
            "sha256": manifest_digest(calibration_records),
            "counts": count_by_label(calibration_records),
        },
        "evaluation": {
            "path": evaluation_path.name,
            "sha256": manifest_digest(evaluation_records),
            "counts": count_by_label(evaluation_records),
        },
    }

    write_json(metadata_path, metadata, args.overwrite)

    print(f"Calibration: {len(calibration_records)} images")
    print(f"Evaluation: {len(evaluation_records)} images")
    print(f"Manifests written to: {args.output_dir}")


if __name__ == "__main__":
    main()