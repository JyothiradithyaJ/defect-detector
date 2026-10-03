"""Normal-reference patch memory for few-normal-shot anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import torch
import torch.nn.functional as functional

from app.core.clip_encoder import MODEL_NAME, PRETRAINED_CHECKPOINT
from PIL import Image

from app.core.clip_encoder import CLIPEncoder, EMBEDDING_DIM, ImageEmbeddings


@dataclass(frozen=True)
class ReferenceScore:
    image_score: torch.Tensor
    patch_score: torch.Tensor


class NormalReferenceBank:
    """Category-specific memory of normal MVTec train/good patch features."""

    def __init__(
        self,
        encoder: CLIPEncoder,
        data_root: Path,
        manifest_path: Path,
        cache_dir: Path,
        max_images_per_category: int | None = None,
        chunk_size: int = 4096,
        top_fraction: float = 0.10,
    ) -> None:
        self.encoder = encoder
        self.data_root = data_root
        self.manifest_path = manifest_path
        self.cache_dir = cache_dir
        self.max_images_per_category = max_images_per_category
        self.chunk_size = chunk_size
        if not 0.0 < top_fraction <= 1.0:
            raise ValueError("top_fraction must be > 0 and <= 1.")
        self.top_fraction = top_fraction
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._banks: dict[str, tuple[torch.Tensor, ...]] = {}
        self._manifest_hash = sha256(manifest_path.read_bytes()).hexdigest()

    def _path(self, category: str) -> Path:
        cache_identity = {
            "manifest_hash": self._manifest_hash,
            "category": category,
            "model_name": MODEL_NAME,
            "pretrained_checkpoint": PRETRAINED_CHECKPOINT,
            "embedding_dim": EMBEDDING_DIM,
            "projection_shape": tuple(self.encoder.model.visual.proj.shape),
            "max_images_per_category": self.max_images_per_category,
            "top_fraction": self.top_fraction,
        }
        key = sha256(
            json.dumps(cache_identity, sort_keys=True, default=str).encode()
        ).hexdigest()[:20]
        return self.cache_dir / f"{category}_{key}.pt"

    def _load_records(self, category: str) -> list[dict[str, object]]:
        records = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        selected = [
            r for r in records
            if str(r["category"]) == category and int(r["label"]) == 0
        ]
        if not selected:
            raise ValueError(f"No normal reference images found for '{category}'.")
        selected = sorted(selected, key=lambda r: str(r["image_path"]))
        if self.max_images_per_category is not None:
            selected = selected[:self.max_images_per_category]
        return selected

    def _build(self, category: str) -> tuple[torch.Tensor, ...]:
        layer_sets: list[list[torch.Tensor]] = []
        records = self._load_records(category)

        for index, record in enumerate(records, start=1):
            image_path = self.data_root / str(record["image_path"])
            if not image_path.is_file():
                raise FileNotFoundError(f"Reference image not found: {image_path}")
            with Image.open(image_path) as image:
                embeddings = self.encoder.encode_image(
                    self.encoder.prepare_image(image)
                )
            layers = embeddings.patch_embeddings_by_layer or (embeddings.patch_embeddings,)
            if not layer_sets:
                layer_sets = [[] for _ in layers]
            if len(layer_sets) != len(layers):
                raise RuntimeError("Reference layer count changed between images.")
            for layer_index, layer in enumerate(layers):
                layer_sets[layer_index].append(layer.squeeze(0).cpu().half())
            if index % 50 == 0 or index == len(records):
                print(f"Reference bank {category}: {index}/{len(records)} images")

        banks = tuple(
            functional.normalize(torch.cat(parts).float(), dim=-1).half()
            for parts in layer_sets
        )
        torch.save({"patch_banks": banks, "images": len(records)}, self._path(category))
        return banks

    def get(self, category: str) -> tuple[torch.Tensor, ...]:
        if category not in self._banks:
            path = self._path(category)
            if path.exists():
                payload = torch.load(path, map_location="cpu", weights_only=True)
                banks = tuple(payload["patch_banks"])
            else:
                banks = self._build(category)
            if not banks or any(
                bank.ndim != 2 or bank.shape[1] != EMBEDDING_DIM for bank in banks
            ):
                raise RuntimeError("Invalid reference bank.")
            self._banks[category] = banks
        return self._banks[category]

    @torch.inference_mode()
    def score(self, category: str, image_embeddings: ImageEmbeddings) -> ReferenceScore:
        banks = self.get(category)
        query_layers = image_embeddings.patch_embeddings_by_layer or (
            image_embeddings.patch_embeddings,
        )
        if len(banks) != len(query_layers):
            raise RuntimeError(
                f"Reference/query layer mismatch: {len(banks)} vs {len(query_layers)}"
            )

        layer_maps = []
        for query, bank_cpu in zip(query_layers, banks):
            query = query.to(self.encoder.device)
            bank = bank_cpu.to(self.encoder.device, dtype=query.dtype)
            best = torch.full(
                (query.shape[0], query.shape[1]),
                -1.0,
                device=self.encoder.device,
                dtype=query.dtype,
            )
            for start in range(0, bank.shape[0], self.chunk_size):
                chunk = bank[start:start + self.chunk_size]
                similarity = torch.matmul(query, chunk.T)
                best = torch.maximum(best, similarity.max(dim=-1).values)
            layer_maps.append((1.0 - best).clamp_min(0.0))

        patch_score = torch.stack(layer_maps).mean(dim=0)
        flat = patch_score.flatten(start_dim=1)
        count = max(1, int(flat.shape[1] * self.top_fraction))
        image_score = flat.topk(count, dim=1).values.mean(dim=1)
        return ReferenceScore(image_score=image_score, patch_score=patch_score)
