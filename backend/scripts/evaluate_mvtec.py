"""Evaluate the pure zero-shot CLIP anomaly detector on MVTec AD."""

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
from app.core.clip_encoder import MODEL_NAME, PRETRAINED_CHECKPOINT, CLIPEncoder
from app.core.evaluation import (
    expected_calibration_error,
    image_auroc,
    pixel_aupro,
    tensor_heatmap_to_numpy,
)
from app.core.fusion import fuse_scores
from app.core.prompts import PromptBank

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "mvtec_ad"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "evaluation.json"
DEFAULT_CALIBRATION = PROJECT_ROOT / "backend" / "config" / "calibration.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "evaluations"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the pure zero-shot CLIP anomaly detector."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--temperature-file", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--categories", nargs="*", default=None)
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not records:
        raise ValueError("Manifest contains no records.")
    return records


def resize_heatmap(heatmap: torch.Tensor, height: int, width: int) -> np.ndarray:
    resized = functional.interpolate(
        heatmap.unsqueeze(1),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return tensor_heatmap_to_numpy(resized.squeeze(1))


@torch.inference_mode()
def score_record(
    record: dict[str, object],
    data_root: Path,
    encoder: CLIPEncoder,
    prompt_bank: PromptBank,
    calibration,
) -> dict[str, object]:
    category = str(record["category"])
    label = int(record["label"])
    image_path = data_root / str(record["image_path"])
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as image:
        original = image.convert("RGB")
        embeddings = encoder.encode_image(encoder.prepare_image(original))

    prompts = prompt_bank.get(category)
    language = fuse_scores(embeddings, prompts)
    defect_logit = language.defect_logit
    probability = calibrate_probability(defect_logit, calibration.temperature)

    width, height = original.size
    if label:
        mask_path = data_root / str(record["mask_path"])
        if not mask_path.is_file():
            raise FileNotFoundError(f"Ground-truth mask not found: {mask_path}")
        with Image.open(mask_path) as mask:
            gt = np.asarray(
                mask.convert("L").resize(
                    (width, height),
                    Image.Resampling.NEAREST,
                ),
                dtype=np.uint8,
            ) > 0
    else:
        gt = np.zeros((height, width), dtype=bool)

    heatmap = resize_heatmap(language.heatmap, height, width)

    return {
        "image_id": str(record["image_id"]),
        "category": category,
        "defect_type": str(record["defect_type"]),
        "label": label,
        "defect_logit": float(defect_logit.item()),
        "defect_probability": float(probability.item()),
        "heatmap": heatmap,
        "ground_truth_mask": gt,
    }


def evaluate_category(
    category: str,
    records: list[dict[str, object]],
) -> dict[str, object]:
    labels = [int(r["label"]) for r in records]
    logits = [float(r["defect_logit"]) for r in records]
    probabilities = [float(r["defect_probability"]) for r in records]
    heatmaps = np.stack([r["heatmap"] for r in records])
    masks = np.stack([r["ground_truth_mask"] for r in records])

    result = {
        "category": category,
        "images": len(records),
        "good_images": sum(label == 0 for label in labels),
        "defective_images": sum(label == 1 for label in labels),
        "image_auroc": image_auroc(labels, logits),
        "pixel_aupro": pixel_aupro(heatmaps, masks),
        "expected_calibration_error": expected_calibration_error(
            probabilities,
            labels,
        ),
    }

    good = [record for record in records if int(record["label"]) == 0]
    defect_types = sorted(
        {
            str(record["defect_type"])
            for record in records
            if int(record["label"]) == 1
        }
    )
    per_defect_type = {}
    for defect_type in defect_types:
        subset = good + [
            record
            for record in records
            if int(record["label"]) == 1
            and str(record["defect_type"]) == defect_type
        ]
        subset_labels = [int(record["label"]) for record in subset]
        subset_logits = [float(record["defect_logit"]) for record in subset]
        subset_heatmaps = np.stack([record["heatmap"] for record in subset])
        subset_masks = np.stack([record["ground_truth_mask"] for record in subset])
        per_defect_type[defect_type] = {
            "images": len(subset),
            "defective_images": len(subset) - len(good),
            "image_auroc": image_auroc(subset_labels, subset_logits),
            "pixel_aupro": pixel_aupro(subset_heatmaps, subset_masks),
        }
    result["defect_types"] = per_defect_type
    return result


def main() -> None:
    args = parse_arguments()
    records = load_records(args.split_manifest)
    if args.categories is not None:
        selected = set(args.categories)
        records = [r for r in records if str(r["category"]) in selected]
        if not records:
            raise ValueError("No records match the requested categories.")

    calibration = load_calibration(args.temperature_file)
    encoder = CLIPEncoder(device=args.device)
    prompt_bank = PromptBank(encoder)

    results = []
    all_labels, all_logits, all_probabilities = [], [], []
    categories = sorted({str(r["category"]) for r in records})
    processed = 0

    for category in categories:
        scored = []
        for record in [r for r in records if str(r["category"]) == category]:
            scored.append(
                score_record(
                    record,
                    args.data_root,
                    encoder,
                    prompt_bank,
                    calibration,
                )
            )
            processed += 1
            if processed % 50 == 0 or processed == len(records):
                print(f"Evaluated {processed}/{len(records)} images")
        results.append(evaluate_category(category, scored))
        all_labels.extend(r["label"] for r in scored)
        all_logits.extend(r["defect_logit"] for r in scored)
        all_probabilities.extend(r["defect_probability"] for r in scored)

    summary = {
        "images": len(all_labels),
        "overall_image_auroc": image_auroc(all_labels, all_logits),
        "macro_image_auroc": float(
            np.mean([r["image_auroc"] for r in results])
        ),
        "macro_pixel_aupro": float(
            np.mean([r["pixel_aupro"] for r in results])
        ),
        "overall_expected_calibration_error": expected_calibration_error(
            all_probabilities,
            all_labels,
        ),
        "macro_expected_calibration_error": float(
            np.mean(
                [r["expected_calibration_error"] for r in results]
            )
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.split_manifest.resolve()),
        "model_name": MODEL_NAME,
        "pretrained_checkpoint": PRETRAINED_CHECKPOINT,
        "device": str(encoder.device),
        "score_mode": "language",
        "global_weight": 0.35,
        "local_weight": 0.65,
        "temperature": calibration.temperature,
        "prompt_cache_key": prompt_bank.cache_key,
        "heatmap_method": "multi-layer CLIP language map",
        "pixel_aupro_protocol": "validated MVTec-compatible AUPRO with max FPR 0.30",
        "summary": summary,
        "categories": results,
    }

    json_path = args.output_dir / "results.json"
    csv_path = args.output_dir / "results.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "category",
                "images",
                "good_images",
                "defective_images",
                "image_auroc",
                "pixel_aupro",
                "expected_calibration_error",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nEvaluation complete")
    print(f"Overall image AUROC: {summary['overall_image_auroc']:.4f}")
    print(f"Macro image AUROC: {summary['macro_image_auroc']:.4f}")
    print(f"Macro pixel AU-PRO: {summary['macro_pixel_aupro']:.4f}")
    print(
        "Overall ECE: "
        f"{summary['overall_expected_calibration_error']:.4f}"
    )
    print(f"JSON report: {json_path}")
    print(f"CSV report: {csv_path}")


if __name__ == "__main__":
    main()
