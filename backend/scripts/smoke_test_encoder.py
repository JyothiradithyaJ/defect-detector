"""Run a shape and freeze check for the CLIP encoder."""

import argparse
from pathlib import Path
import sys


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import open_clip
import torch
from PIL import Image

from app.core.clip_encoder import (
    EMBEDDING_DIM,
    PATCH_COUNT,
    CLIPEncoder,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to a test image.")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    encoder = CLIPEncoder()

    with Image.open(args.image) as image:
        image_batch = encoder.prepare_image(image)

    embeddings = encoder.encode_image(image_batch)

    assert embeddings.global_embedding.shape == (1, EMBEDDING_DIM)
    assert embeddings.patch_embeddings.shape == (1, PATCH_COUNT, EMBEDDING_DIM)
    assert not any(parameter.requires_grad for parameter in encoder.model.parameters())

    print(f"Model: {encoder.model.__class__.__name__}")
    print(f"OpenCLIP: {open_clip.__version__}")
    print(f"Device: {encoder.device}")
    print(f"Global embedding: {tuple(embeddings.global_embedding.shape)}")
    print(f"Patch embeddings: {tuple(embeddings.patch_embeddings.shape)}")
    print("Backbone frozen: True")


if __name__ == "__main__":
    main()
