"""Evaluate frozen CLIP fusion on the fixed MVTec evaluation manifest."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from app.core.calibration import calibrate_probability, load_calibration
from app.core.clip_encoder import (
    MODEL_NAME,
    PRETRAINED_CHECKPOINT,
    CLIPEncoder,
)
from app.core.evaluation import (
    expected_calibration_error,
    image_auroc,
    pixel_aupro,
    tensor_heatmap_to_numpy,
)
from app.core.fusion import ALPHA, fuse_scores
from app.core.prompts import PromptBank


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "mvtec_ad"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "evaluation.json"
DEFAULT_CALIBRATION = PROJECT_ROOT / "backend" / "config" / "calibration.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "evaluations"


def parse_arguments() -> argparse.Namespace:
    """Read evaluator settings."""
    parser = argparse.ArgumentParser(
        description="Evaluate the CLIP defect-detection pipeline on MVTec AD."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--temperature-file",
        type=Path,
        default=DEFAULT_CALIBRATION,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Optional list of categories, for example: bottle cable zipper",
    )
    return parser.parse_args()


def load_records(manifest_path: Path) -> list[dict[str, object]]:
    """Load and validate records from an evaluation manifest."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Evaluation manifest not found: {manifest_path}")

    records = json.loads(manifest_path.read_text(encoding="utf-8"))

    if not records:
        raise ValueError("Evaluation manifest contains no records.")

    return records


def resize_heatmap(
    heatmap: torch.Tensor,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    """Resize one 7x7 CLIP heatmap to the original image/mask resolution."""
    heatmap_tensor = heatmap.unsqueeze(1)

    resized = functional.interpolate(
        heatmap_tensor,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )

    return tensor_heatmap_to_numpy(resized.squeeze(1))


def score_record(
    record: dict[str, object],
    data_root: Path,
    encoder: CLIPEncoder,
    prompt_bank: PromptBank,
    temperature: float,
) -> dict[str, object]:
    """Run encoder, prompts, fusion, calibration, and heatmap resizing."""
    image_path = data_root / str(record["image_path"])
    category = str(record["category"])
    label = int(record["label"])

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as image:
        original_image = image.convert("RGB")
        image_batch = encoder.prepare_image(original_image)

    image_embeddings = encoder.encode_image(image_batch)
    prompt_embeddings = prompt_bank.get(category)
    fusion_result = fuse_scores(image_embeddings, prompt_embeddings)

    defect_logit = float(fusion_result.defect_logit.item())
    probability = float(
        calibrate_probability(
            fusion_result.defect_logit,
            temperature,
        ).item()
    )

    width, height = original_image.size

    if label == 1:
        mask_path = data_root / str(record["mask_path"])

        if not mask_path.is_file():
            raise FileNotFoundError(f"Ground-truth mask not found: {mask_path}")

        with Image.open(mask_path) as mask:
            ground_truth_mask = np.asarray(
                mask.convert("L").resize(
                    (width, height),
                    Image.Resampling.NEAREST,
                ),
                dtype=np.uint8,
            ) > 0
    else:
        ground_truth_mask = np.zeros((height, width), dtype=bool)

    heatmap = resize_heatmap(
        fusion_result.heatmap,
        target_height=height,
        target_width=width,
    )

    return {
        "image_id": str(record["image_id"]),
        "category": category,
        "defect_type": str(record["defect_type"]),
        "label": label,
        "defect_logit": defect_logit,
        "defect_probability": probability,
        "heatmap": heatmap,
        "ground_truth_mask": ground_truth_mask,
    }


def evaluate_category(
    category: str,
    records: list[dict[str, object]],
) -> dict[str, object]:
    """Calculate image and pixel metrics for one MVTec category."""
    labels = [int(record["label"]) for record in records]
    logits = [float(record["defect_logit"]) for record in records]
    probabilities = [
        float(record["defect_probability"])
        for record in records
    ]

    heatmaps = np.stack([record["heatmap"] for record in records])
    masks = np.stack([record["ground_truth_mask"] for record in records])

    return {
        "category": category,
        "images": len(records),
        "good_images": int(sum(label == 0 for label in labels)),
        "defective_images": int(sum(label == 1 for label in labels)),
        "image_auroc": image_auroc(labels, logits),
        "pixel_aupro": pixel_aupro(heatmaps, masks),
        "expected_calibration_error": expected_calibration_error(
            probabilities,
            labels,
        ),
    }


def write_csv(results: list[dict[str, object]], path: Path) -> None:
    """Write category metrics as a spreadsheet-friendly CSV file."""
    fieldnames = [
        "category",
        "images",
        "good_images",
        "defective_images",
        "image_auroc",
        "pixel_aupro",
        "expected_calibration_error",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    """Run the evaluation split and save JSON and CSV reports."""
    args = parse_arguments()
    records = load_records(args.split_manifest)

    if args.categories is not None:
        selected_categories = set(args.categories)
        records = [
            record
            for record in records
            if str(record["category"]) in selected_categories
        ]

        if not records:
            raise ValueError("No records match the requested categories.")

    calibration = load_calibration(args.temperature_file)
    encoder = CLIPEncoder(device=args.device)
    prompt_bank = PromptBank(encoder)

    categories = sorted({str(record["category"]) for record in records})
    category_results = []
    all_labels: list[int] = []
    all_logits: list[float] = []
    all_probabilities: list[float] = []
    processed = 0

    # Score and evaluate one category at a time. This releases high-resolution
    # heatmaps before moving to the next category, keeping CPU memory bounded.
    for category in categories:
        category_source_records = [
            record for record in records if str(record["category"]) == category
        ]
        category_scored_records = []

        for record in category_source_records:
            category_scored_records.append(
                score_record(
                    record=record,
                    data_root=args.data_root,
                    encoder=encoder,
                    prompt_bank=prompt_bank,
                    temperature=calibration.temperature,
                )
            )
            processed += 1

            if processed % 50 == 0 or processed == len(records):
                print(f"Evaluated {processed}/{len(records)} images")

        category_results.append(
            evaluate_category(category, category_scored_records)
        )

        all_labels.extend(
            int(record["label"]) for record in category_scored_records
        )
        all_logits.extend(
            float(record["defect_logit"])
            for record in category_scored_records
        )
        all_probabilities.extend(
            float(record["defect_probability"])
            for record in category_scored_records
        )

    macro_image_auroc = float(
        np.mean([result["image_auroc"] for result in category_results])
    )
    macro_pixel_aupro = float(
        np.mean([result["pixel_aupro"] for result in category_results])
    )
    macro_ece = float(
        np.mean(
            [
                result["expected_calibration_error"]
                for result in category_results
            ]
        )
    )

    summary = {
        "images": len(all_labels),
        "overall_image_auroc": image_auroc(all_labels, all_logits),
        "overall_expected_calibration_error": expected_calibration_error(
            all_probabilities,
            all_labels,
        ),
        "macro_image_auroc": macro_image_auroc,
        "macro_pixel_aupro": macro_pixel_aupro,
        "macro_expected_calibration_error": macro_ece,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.output_dir / "results.json"
    csv_path = args.output_dir / "results.csv"

    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.split_manifest.resolve()),
        "model_name": MODEL_NAME,
        "pretrained_checkpoint": PRETRAINED_CHECKPOINT,
        "device": str(encoder.device),
        "alpha": ALPHA,
        "temperature": calibration.temperature,
        "calibration_manifest_hash": calibration.calibration_manifest_hash,
        "prompt_cache_key": prompt_bank.cache_key,
        "heatmap_method": (
            "7x7 anomalous-minus-normal CLIP patch grid, "
            "bilinearly resized to original image resolution"
        ),
        "pixel_aupro_note": (
            "Approximate AU-PRO using 200 thresholds and "
            "a maximum false-positive rate of 0.30."
        ),
        "summary": summary,
        "categories": category_results,
    }

    json_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    write_csv(category_results, csv_path)

    print("\nEvaluation complete")
    print(f"Overall image AUROC: {summary['overall_image_auroc']:.4f}")
    print(f"Macro image AUROC: {summary['macro_image_auroc']:.4f}")
    print(f"Macro pixel AU-PRO: {summary['macro_pixel_aupro']:.4f}")
    print(f"Overall ECE: {summary['overall_expected_calibration_error']:.4f}")
    print(f"JSON report: {json_path}")
    print(f"CSV report: {csv_path}")


if __name__ == "__main__":
    main()
