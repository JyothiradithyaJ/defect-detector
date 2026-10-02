

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as functional

from app.core.clip_encoder import (
    EMBEDDING_DIM,
    MODEL_NAME,
    PRETRAINED_CHECKPOINT,
    CLIPEncoder,
)


NORMAL_TEMPLATES = (
    "a photo of a flawless {object}",
    "an undamaged {object}",
)

ANOMALOUS_TEMPLATES = (
    "a photo of a damaged {object}",
    "a {object} with a defect",
)

MVTEC_OBJECT_NAMES = {
    "bottle": "bottle",
    "cable": "cable",
    "capsule": "capsule",
    "carpet": "carpet",
    "grid": "metal grid",
    "hazelnut": "hazelnut",
    "leather": "leather",
    "metal_nut": "metal nut",
    "pill": "pill",
    "screw": "screw",
    "tile": "tile",
    "toothbrush": "toothbrush",
    "transistor": "transistor",
    "wood": "wood",
    "zipper": "zipper",
}

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "prompt_cache"


@dataclass(frozen=True)
class PromptEmbeddings:

    normal: torch.Tensor      
    anomalous: torch.Tensor   


def _hash_value(value: object) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return sha256(serialized.encode("utf-8")).hexdigest()


class PromptBank:

    def __init__(
        self,
        encoder: CLIPEncoder,
        cache_dir: Path = DEFAULT_CACHE_DIR,
    ) -> None:
        self.encoder = encoder
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, PromptEmbeddings] = {}

        self.templates_hash = _hash_value(
            {
                "normal": NORMAL_TEMPLATES,
                "anomalous": ANOMALOUS_TEMPLATES,
            }
        )
        self.object_names_hash = _hash_value(MVTEC_OBJECT_NAMES)
        self.cache_key = _hash_value(
            {
                "model_name": MODEL_NAME,
                "pretrained_checkpoint": PRETRAINED_CHECKPOINT,
                "open_clip_version": open_clip.__version__,
                "templates_hash": self.templates_hash,
                "object_names_hash": self.object_names_hash,
            }
        )

    def get(self, category: str) -> PromptEmbeddings:
        if category not in MVTEC_OBJECT_NAMES:
            available = ", ".join(sorted(MVTEC_OBJECT_NAMES))
            raise ValueError(
                f"Unknown MVTec category '{category}'. Available: {available}"
            )

        if category in self._memory_cache:
            return self._memory_cache[category]

        cached_embeddings = self._load_from_disk(category)
        if cached_embeddings is not None:
            self._memory_cache[category] = cached_embeddings
            return cached_embeddings

        embeddings = self._create_embeddings(category)
        self._save_to_disk(category, embeddings)
        self._memory_cache[category] = embeddings
        return embeddings

    def _create_embeddings(self, category: str) -> PromptEmbeddings:
        object_name = MVTEC_OBJECT_NAMES[category]

        normal_prompts = [
            template.format(object=object_name)
            for template in NORMAL_TEMPLATES
        ]
        anomalous_prompts = [
            template.format(object=object_name)
            for template in ANOMALOUS_TEMPLATES
        ]

        normal = self._mean_normalized_embedding(normal_prompts)
        anomalous = self._mean_normalized_embedding(anomalous_prompts)

        return PromptEmbeddings(normal=normal, anomalous=anomalous)

    def _mean_normalized_embedding(self, prompts: list[str]) -> torch.Tensor:
        embeddings = self.encoder.encode_text(prompts)
        mean_embedding = embeddings.mean(dim=0)

        return functional.normalize(mean_embedding, dim=0)

    def _cache_paths(self, category: str) -> tuple[Path, Path]:
        stem = f"{category}_{self.cache_key}"
        return (
            self.cache_dir / f"{stem}.pt",
            self.cache_dir / f"{stem}.json",
        )

    def _load_from_disk(self, category: str) -> PromptEmbeddings | None:
        tensor_path, metadata_path = self._cache_paths(category)

        if not tensor_path.exists() or not metadata_path.exists():
            return None

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            if metadata.get("cache_key") != self.cache_key:
                return None

            payload = torch.load(
                tensor_path,
                map_location=self.encoder.device,
                weights_only=True,
            )

            normal = payload["normal"]
            anomalous = payload["anomalous"]

            if normal.shape != (EMBEDDING_DIM,):
                return None

            if anomalous.shape != (EMBEDDING_DIM,):
                return None

            return PromptEmbeddings(normal=normal, anomalous=anomalous)

        except (KeyError, OSError, RuntimeError, json.JSONDecodeError):
            return None

    def _save_to_disk(
        self,
        category: str,
        embeddings: PromptEmbeddings,
    ) -> None:
        tensor_path, metadata_path = self._cache_paths(category)

        torch.save(
            {
                "normal": embeddings.normal.cpu(),
                "anomalous": embeddings.anomalous.cpu(),
            },
            tensor_path,
        )

        metadata = {
            "category": category,
            "object_name": MVTEC_OBJECT_NAMES[category],
            "model_name": MODEL_NAME,
            "pretrained_checkpoint": PRETRAINED_CHECKPOINT,
            "open_clip_version": open_clip.__version__,
            "templates_hash": self.templates_hash,
            "object_names_hash": self.object_names_hash,
            "cache_key": self.cache_key,
        }

        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )