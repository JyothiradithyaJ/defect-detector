"""Fit one temperature for the pure zero-shot CLIP anomaly score."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import torch
import torch.nn.functional as functional
import numpy as np
from PIL import Image

from app.core.calibration import CalibrationConfig, save_calibration

if TYPE_CHECKING:
    from app.core.clip_encoder import CLIPEncoder
    from app.core.prompts import PromptBank

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "mvtec_ad"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "calibration.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "backend" / "config" / "calibration.json"
DEFAULT_DIAGNOSTIC_DIR = PROJECT_ROOT


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit temperature scaling for the language-only anomaly score."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--diagnostic-dir",
        type=Path,
        default=DEFAULT_DIAGNOSTIC_DIR,
        help=(
            "Directory for raw calibration-logit diagnostics "
            "(default: repository root)."
        ),
    )
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Keep this import lazy: fitting unit tests only need the optimizer and
    # should not require an installed/downloaded OpenCLIP model.
    from app.core.fusion import fuse_scores

    scores, labels, components = [], [], []

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
        components.append(
            torch.stack(
                (
                    result.anomalous_global - result.normal_global,
                    result.anomalous_local - result.normal_local,
                ),
                dim=1,
            ).cpu()
        )
        labels.append(int(record["label"]))

        if index % 50 == 0 or index == len(records):
            print(f"Encoded {index}/{len(records)} calibration images")

    return (
        torch.cat(scores).double(),
        torch.tensor(labels, dtype=torch.float64),
        torch.cat(components).double(),
    )


def fit_temperature(
    defect_logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float, float]:
    """Fit a positive temperature and an intercept by minimizing binary NLL.

    Raw CLIP similarity differences are scores, not calibrated log odds.  A
    temperature alone can only soften or sharpen probabilities around 0.5;
    it cannot account for the class prevalence of a held-out split.  The
    intercept supplies that missing degree of freedom while the temperature
    preserves the detector's score ordering.
    """
    logits = defect_logits.detach().clone().double()
    targets = labels.detach().clone().double()
    if logits.ndim != 1 or targets.ndim != 1 or logits.shape != targets.shape:
        raise ValueError("Logits and labels must be matching one-dimensional tensors.")
    if not torch.isfinite(logits).all() or not torch.isfinite(targets).all():
        raise ValueError("Logits and labels must be finite.")
    if not torch.all((targets == 0) | (targets == 1)):
        raise ValueError("Labels must be binary (0 or 1).")

    log_temperature = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    bias = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [log_temperature, bias],
        lr=0.1,
        max_iter=200,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        # No arbitrary calibration bounds: a boundary solution is evidence
        # about the data, not a valid fitted parameter.  float64 keeps the
        # exponent stable for the score scales produced by CLIP similarities.
        temperature = log_temperature.exp()
        loss = functional.binary_cross_entropy_with_logits(
            logits / temperature + bias,
            targets,
        )
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.inference_mode():
        temperature = float(log_temperature.exp())
        fitted_bias = float(bias)
        nll = float(
            functional.binary_cross_entropy_with_logits(
                logits / temperature + fitted_bias,
                targets,
            )
        )

    if not all(np.isfinite(value) for value in (temperature, fitted_bias, nll)):
        raise RuntimeError("Calibration optimization produced non-finite values.")
    return temperature, fitted_bias, nll


def save_diagnostics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    components: torch.Tensor,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """Save raw calibration inputs for standalone investigation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logits_path = output_dir / "calib_logits.npy"
    labels_path = output_dir / "calib_labels.npy"
    components_path = output_dir / "calib_score_components.npy"

    np.save(logits_path, logits.detach().cpu().numpy())
    np.save(labels_path, labels.detach().cpu().numpy())
    np.save(components_path, components.detach().cpu().numpy())

    return logits_path, labels_path, components_path


def main() -> None:
    args = parse_arguments()
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    records = load_records(args.manifest)
    from app.core.clip_encoder import MODEL_NAME, PRETRAINED_CHECKPOINT, CLIPEncoder
    from app.core.prompts import PromptBank

    encoder = CLIPEncoder(device=args.device)
    prompt_bank = PromptBank(encoder)
    logits, labels, components = collect_scores(
        records,
        args.data_root,
        encoder,
        prompt_bank,
    )
    logits_path, labels_path, components_path = save_diagnostics(
        logits,
        labels,
        components,
        args.diagnostic_dir,
    )
    temperature, bias, nll = fit_temperature(logits, labels)

    config = CalibrationConfig(
        temperature=temperature,
        bias=bias,
        model_name=MODEL_NAME,
        pretrained_checkpoint=PRETRAINED_CHECKPOINT,
        calibration_manifest_hash=file_sha256(args.manifest),
        prompt_cache_key=prompt_bank.cache_key,
        fitted_at=datetime.now(timezone.utc).isoformat(),
        calibration_nll=nll,
    )
    save_calibration(config, args.output)

    print(f"Temperature: {temperature:.6f}")
    print(f"Calibration bias: {bias:.6f}")
    print(f"Calibration NLL: {nll:.6f}")
    print(f"Saved: {args.output}")
    print(f"Diagnostic logits: {logits_path}")
    print(f"Diagnostic labels: {labels_path}")
    print(f"Diagnostic score components: {components_path}")


if __name__ == "__main__":
    main()
