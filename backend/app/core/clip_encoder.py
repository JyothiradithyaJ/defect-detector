"""Frozen OpenCLIP encoder with multi-layer patch features."""

from dataclasses import dataclass
from typing import Callable

import open_clip
import torch
import torch.nn.functional as functional
from PIL import Image

MODEL_NAME = "ViT-B-32-quickgelu"
PRETRAINED_CHECKPOINT = "openai"
EMBEDDING_DIM = 512
PATCH_GRID_SIZE = 7
PATCH_COUNT = PATCH_GRID_SIZE * PATCH_GRID_SIZE
INTERMEDIATE_LAYER_COUNT = 3


@dataclass(frozen=True)
class ImageEmbeddings:
    """Global and multi-layer spatial CLIP embeddings."""

    global_embedding: torch.Tensor
    patch_embeddings: torch.Tensor
    patch_embeddings_by_layer: tuple[torch.Tensor, ...] = ()


class CLIPEncoder:
    """Load a frozen OpenAI CLIP ViT-B/32 backbone."""

    def __init__(self, device: str | torch.device | None = None) -> None:
        self.device = torch.device(device or "cpu")
        model, _, preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME,
            pretrained=PRETRAINED_CHECKPOINT,
            device=self.device,
        )
        self.model = model.eval()
        self.preprocess: Callable[[Image.Image], torch.Tensor] = preprocess
        self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def prepare_image(self, image: Image.Image) -> torch.Tensor:
        return self.preprocess(image.convert("RGB")).unsqueeze(0)

    @torch.inference_mode()
    def encode_image(self, image_batch: torch.Tensor) -> ImageEmbeddings:
        if image_batch.ndim != 4:
            raise ValueError(
                "Expected image_batch with shape [batch, channels, height, width]."
            )

        image_batch = image_batch.to(self.device)
        outputs = self.model.forward_intermediates(
            image=image_batch,
            image_indices=INTERMEDIATE_LAYER_COUNT,
            normalize=True,
            normalize_intermediates=True,
            image_output_fmt="NLC",
        )

        global_embedding = outputs["image_features"]
        intermediates = outputs["image_intermediates"]
        projection = self.model.visual.proj
        if projection is None:
            raise RuntimeError("ViT-B/32 visual projection is unavailable.")

        projected_layers = tuple(
            functional.normalize(layer @ projection, dim=-1)
            for layer in intermediates
        )
        if not projected_layers:
            raise RuntimeError("CLIP returned no intermediate image features.")

        patch_embeddings = projected_layers[-1]

        if global_embedding.shape[-1] != EMBEDDING_DIM:
            raise RuntimeError(
                f"Expected {EMBEDDING_DIM}-d global embedding, "
                f"got {global_embedding.shape[-1]}."
            )

        if patch_embeddings.shape[1:] != (PATCH_COUNT, EMBEDDING_DIM):
            raise RuntimeError(
                f"Expected patch embeddings [batch, {PATCH_COUNT}, {EMBEDDING_DIM}], "
                f"got {tuple(patch_embeddings.shape)}."
            )

        return ImageEmbeddings(
            global_embedding=global_embedding,
            patch_embeddings=patch_embeddings,
            patch_embeddings_by_layer=projected_layers,
        )

    @torch.inference_mode()
    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        if not prompts:
            raise ValueError("At least one text prompt is required.")
        tokens = self.tokenizer(prompts).to(self.device)
        return self.model.encode_text(tokens, normalize=True)
