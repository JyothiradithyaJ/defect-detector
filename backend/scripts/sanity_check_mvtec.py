"""Run the current CLIP prompt-and-fusion pipeline on one MVTec image."""

from __future__ import annotations
import argparse
from pathlib import Path
import sys


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from PIL import Image

from app.core.clip_encoder import CLIPEncoder
from app.core.fusion import fuse_scores
from app.core.heatmap import heatmap_to_overlay
from app.core.prompts import MVTEC_OBJECT_NAMES, PromptBank


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score one MVTec image with the current CLIP pipeline."
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to a MVTec image.",
    )
    parser.add_argument(
        "--category",
        required=True,
        choices=sorted(MVTEC_OBJECT_NAMES),
        help="MVTec product category for the image.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts"),
        help="Directory where the heatmap overlay is saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")

    encoder = CLIPEncoder()
    prompt_bank = PromptBank(encoder)

    with Image.open(args.image) as image:
        original_image = image.convert("RGB")
        image_batch = encoder.prepare_image(original_image)

    image_embeddings = encoder.encode_image(image_batch)
    prompt_embeddings = prompt_bank.get(args.category)
    result = fuse_scores(image_embeddings, prompt_embeddings)
    args.output.mkdir(parents=True, exist_ok=True)

    overlay = heatmap_to_overlay(original_image, result.heatmap)

    defect_type = args.image.parent.name
    output_path = args.output / (
    f"{args.category}_{defect_type}_{args.image.stem}_heatmap.png"
)
    overlay.save(output_path)
    print(f"Image: {args.image}")
    print(f"Category: {args.category}")
    print(f"Normal global score: {result.normal_global.item():.4f}")
    print(f"Anomalous global score: {result.anomalous_global.item():.4f}")
    print(f"Normal local score: {result.normal_local.item():.4f}")
    print(f"Anomalous local score: {result.anomalous_local.item():.4f}")
    print(f"Normal fused score: {result.normal_fused.item():.4f}")
    print(f"Anomalous fused score: {result.anomalous_fused.item():.4f}")
    print(f"Defect logit: {result.defect_logit.item():.4f}")
    print(f"Heatmap shape: {tuple(result.heatmap.shape)}")
    print(f"Heatmap overlay: {output_path}")


if __name__ == "__main__":
    main()
