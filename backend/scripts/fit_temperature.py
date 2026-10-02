

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
        description="Fit temperature scaling on a calibration manifest."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def file_sha256(path: Path) -> str:

    return sha256(path.read_bytes()).hexdigest()


def load_records(manifest_path: Path) -> list[dict[str, object]]:

    records = json.loads(manifest_path.read_text(encoding="utf-8"))

    labels = {int(record["label"]) for record in records}

    if labels != {0, 1}:
        raise ValueError(
            "Calibration manifest must contain both good (0) "
            "and defective (1) images."
        )

    return records


@torch.inference_mode()
def collect_logits(
    records: list[dict[str, object]],
    data_root: Path,
    encoder: CLIPEncoder,
    prompt_bank: PromptBank,
) -> tuple[torch.Tensor, torch.Tensor]:

    from PIL import Image

    logits = []
    labels = []

    for index, record in enumerate(records, start=1):
        image_path = data_root / str(record["image_path"])
        category = str(record["category"])

        if not image_path.is_file():
            raise FileNotFoundError(f"Calibration image not found: {image_path}")

        with Image.open(image_path) as image:
            image_batch = encoder.prepare_image(image)

        image_embeddings = encoder.encode_image(image_batch)
        prompt_embeddings = prompt_bank.get(category)
        fusion_result = fuse_scores(image_embeddings, prompt_embeddings)

        logits.append(fusion_result.defect_logit.cpu())
        labels.append(int(record["label"]))

        if index % 50 == 0 or index == len(records):
            print(f"Encoded {index}/{len(records)} calibration images")

    return (
        torch.cat(logits).to(encoder.device),
        torch.tensor(labels, dtype=torch.float32, device=encoder.device),
    )


def fit_temperature(
    defect_logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float]:
    """Minimise binary negative log likelihood with LBFGS."""
    # Logits are collected under torch.inference_mode() for efficient CLIP
    # encoding. LBFGS needs ordinary tensors because it differentiates with
    # respect to the temperature parameter while using these values in loss.
    defect_logits = defect_logits.detach().clone()
    labels = labels.detach().clone()

    log_temperature = torch.nn.Parameter(
        torch.tensor(0.0, device=defect_logits.device)
    )

    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=50,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()

        temperature = log_temperature.exp()
        loss = functional.binary_cross_entropy_with_logits(
            defect_logits / temperature,
            labels,
        )

        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.inference_mode():
        temperature = log_temperature.exp().item()
        nll = functional.binary_cross_entropy_with_logits(
            defect_logits / temperature,
            labels,
        ).item()

    return temperature, nll


def main() -> None:
    args = parse_arguments()

    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    records = load_records(args.manifest)

    encoder = CLIPEncoder(device=args.device)
    prompt_bank = PromptBank(encoder)

    defect_logits, labels = collect_logits(
        records=records,
        data_root=args.data_root,
        encoder=encoder,
        prompt_bank=prompt_bank,
    )

    temperature, nll = fit_temperature(defect_logits, labels)

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

    print(f"Fitted temperature: {temperature:.6f}")
    print(f"Calibration NLL: {nll:.6f}")
    print(f"Saved calibration: {args.output}")


if __name__ == "__main__":
    main()
