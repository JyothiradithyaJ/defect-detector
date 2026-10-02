"""Fit language/reference fusion and temperature on the calibration split."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import torch
import torch.nn.functional as functional
from PIL import Image

from app.core.calibration import CalibrationConfig, save_calibration
from app.core.clip_encoder import MODEL_NAME, PRETRAINED_CHECKPOINT, CLIPEncoder
from app.core.fusion import fuse_scores
from app.core.prompts import PromptBank
from app.core.reference_bank import NormalReferenceBank

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "mvtec_ad"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "calibration.json"
DEFAULT_REFERENCE_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "train_reference.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "backend" / "config" / "calibration.json"
DEFAULT_REFERENCE_CACHE = PROJECT_ROOT / "data" / "reference_cache"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit anomaly score calibration.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reference-manifest", type=Path, default=DEFAULT_REFERENCE_MANIFEST)
    parser.add_argument("--reference-cache", type=Path, default=DEFAULT_REFERENCE_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load_records(path: Path) -> list[dict[str, object]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if {int(r["label"]) for r in records} != {0, 1}:
        raise ValueError("Calibration manifest must contain both labels 0 and 1.")
    return records


@torch.inference_mode()
def collect_components(records, data_root, encoder, prompt_bank, reference_bank):
    language_scores, reference_scores, labels = [], [], []
    for index, record in enumerate(records, start=1):
        category = str(record["category"])
        image_path = data_root / str(record["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Calibration image not found: {image_path}")
        with Image.open(image_path) as image:
            embeddings = encoder.encode_image(encoder.prepare_image(image))
        prompts = prompt_bank.get(category)
        reference = reference_bank.score(category, embeddings)
        language = fuse_scores(embeddings, prompts, reference_score=None)
        language_scores.append(language.defect_logit.cpu())
        reference_scores.append(reference.image_score.cpu())
        labels.append(int(record["label"]))
        if index % 50 == 0 or index == len(records):
            print(f"Encoded {index}/{len(records)} calibration images")
    return torch.cat(language_scores), torch.cat(reference_scores), torch.tensor(labels, dtype=torch.float64)


def fit_parameters(language, reference, labels):
    language, reference, labels = language.double(), reference.double(), labels.double()
    log_temperature = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    log_language = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    log_reference = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [log_temperature, log_language, log_reference],
        lr=0.1, max_iter=100, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        language_weight = log_language.exp().clamp(0.05, 5.0)
        reference_weight = log_reference.exp().clamp(0.05, 5.0)
        logits = (language_weight * language + reference_weight * reference) / temperature
        loss = functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.inference_mode():
        temperature = float(log_temperature.exp().clamp(0.05, 20.0))
        language_weight = float(log_language.exp().clamp(0.05, 5.0))
        reference_weight = float(log_reference.exp().clamp(0.05, 5.0))
        logits = (language_weight * language + reference_weight * reference) / temperature
        nll = float(functional.binary_cross_entropy_with_logits(logits, labels))
    return temperature, language_weight, reference_weight, nll


def main() -> None:
    args = parse_arguments()
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    if not args.reference_manifest.is_file():
        raise FileNotFoundError(f"Reference manifest not found: {args.reference_manifest}")

    records = load_records(args.manifest)
    encoder = CLIPEncoder(device=args.device)
    prompt_bank = PromptBank(encoder)
    reference_bank = NormalReferenceBank(
        encoder, args.data_root, args.reference_manifest, args.reference_cache
    )
    language, reference, labels = collect_components(
        records, args.data_root, encoder, prompt_bank, reference_bank
    )
    temperature, language_weight, reference_weight, nll = fit_parameters(
        language, reference, labels
    )

    config = CalibrationConfig(
        temperature=temperature,
        language_weight=language_weight,
        reference_weight=reference_weight,
        model_name=MODEL_NAME,
        pretrained_checkpoint=PRETRAINED_CHECKPOINT,
        calibration_manifest_hash=file_sha256(args.manifest),
        reference_manifest_hash=file_sha256(args.reference_manifest),
        prompt_cache_key=prompt_bank.cache_key,
        fitted_at=datetime.now(timezone.utc).isoformat(),
        calibration_nll=nll,
    )
    save_calibration(config, args.output)
    print(f"Temperature: {temperature:.6f}")
    print(f"Language weight: {language_weight:.6f}")
    print(f"Reference weight: {reference_weight:.6f}")
    print(f"Calibration NLL: {nll:.6f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
