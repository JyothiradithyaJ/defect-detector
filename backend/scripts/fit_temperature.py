"""Fit one temperature for the pure zero-shot CLIP anomaly score."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "mvtec_ad"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "calibration.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "backend" / "config" / "calibration.json"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit temperature scaling for the language-only anomaly score."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
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
def collect_scores(
    records: list[dict[str, object]],
    data_root: Path,
    encoder: CLIPEncoder,
    prompt_bank: PromptBank,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores, labels = [], []

    for index, record in enumerate(records, start=1):
        category = str(record["category"])
        image_path = data_root / str(record["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Calibration image not found: {image_path}")

        with Image.open(image_path) as image:
            embeddings = encoder.encode_image(encoder.prepare_image(image))

        prompts = prompt_bank.get(category)
        result = fuse_scores(embeddings, prompts)
        scores.append(result.defect_logit.cpu())
        labels.append(int(record["label"]))

        if index % 50 == 0 or index == len(records):
            print(f"Encoded {index}/{len(records)} calibration images")

    return (
        torch.cat(scores).double(),
        torch.tensor(labels, dtype=torch.float64),
    )


def fit_temperature(
    defect_logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float]:
    """Fit one positive temperature by minimizing binary NLL."""
    logits = defect_logits.detach().clone().double()
    targets = labels.detach().clone().double()

    log_temperature = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=100,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = functional.binary_cross_entropy_with_logits(
            logits / temperature,
            targets,
        )
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.inference_mode():
        temperature = float(log_temperature.exp().clamp(0.05, 20.0))
        nll = float(
            functional.binary_cross_entropy_with_logits(
                logits / temperature,
                targets,
            )
        )

    return temperature, nll


def main() -> None:
    args = parse_arguments()
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    records = load_records(args.manifest)
    encoder = CLIPEncoder(device=args.device)
    prompt_bank = PromptBank(encoder)
    logits, labels = collect_scores(
        records,
        args.data_root,
        encoder,
        prompt_bank,
    )
    temperature, nll = fit_temperature(logits, labels)

    config = CalibrationConfig(
        temperature=temperature,
        model_name=MODEL_NAME,
        pretrained_checkpoint=PRETRAINED_CHECKPOINT,
        calibration_manifest_hash=file_sha256(args.manifest),
        prompt_cache_key=prompt_bank.cache_key,
        fitted_at=datetime.now(timezone.utc).isoformat(),
        calibration_nll=nll,
    )
    save_calibration(config, args.output)

    print(f"Temperature: {temperature:.6f}")
    print(f"Calibration NLL: {nll:.6f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
