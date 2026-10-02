"""Compositional CLIP prompt ensemble for industrial anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as functional

from app.core.clip_encoder import EMBEDDING_DIM, MODEL_NAME, PRETRAINED_CHECKPOINT, CLIPEncoder

NORMAL_STATES = (
    "{object}",
    "flawless {object}",
    "perfect {object}",
    "unblemished {object}",
    "{object} without flaw",
    "{object} without defect",
    "{object} without damage",
    "normal {object}",
)

ANOMALOUS_STATES = (
    "damaged {object}",
    "broken {object}",
    "abnormal {object}",
    "imperfect {object}",
    "{object} with flaw",
    "{object} with defect",
    "{object} with damage",
    "defective {object}",
)

TEMPLATES = (
    "a cropped photo of the {state}",
    "a cropped photo of a {state}",
    "a close-up photo of a {state}",
    "a close-up photo of the {state}",
    "a bright photo of a {state}",
    "a dark photo of a {state}",
    "a blurry photo of a {state}",
    "a good photo of a {state}",
    "a bad photo of a {state}",
    "a photo of a {state}",
    "a photo of the {state}",
    "a photo of a small {state}",
    "a photo of a large {state}",
    "a photo of the {state} for visual inspection",
    "a photo of a {state} for visual inspection",
    "a photo of the {state} for anomaly detection",
    "a photo of a {state} for anomaly detection",
)

MVTEC_OBJECT_NAMES = {
    "bottle": "bottle", "cable": "cable", "capsule": "capsule",
    "carpet": "carpet", "grid": "metal grid", "hazelnut": "hazelnut",
    "leather": "leather", "metal_nut": "metal nut", "pill": "pill",
    "screw": "screw", "tile": "tile", "toothbrush": "toothbrush",
    "transistor": "transistor", "wood": "wood", "zipper": "zipper",
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
    def __init__(self, encoder: CLIPEncoder, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
        self.encoder = encoder
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, PromptEmbeddings] = {}
        self.templates_hash = _hash_value({
            "normal_states": NORMAL_STATES,
            "anomalous_states": ANOMALOUS_STATES,
            "templates": TEMPLATES,
        })
        self.object_names_hash = _hash_value(MVTEC_OBJECT_NAMES)
        self.cache_key = _hash_value({
            "model_name": MODEL_NAME,
            "pretrained_checkpoint": PRETRAINED_CHECKPOINT,
            "open_clip_version": open_clip.__version__,
            "templates_hash": self.templates_hash,
            "object_names_hash": self.object_names_hash,
        })

    def get(self, category: str) -> PromptEmbeddings:
        if category not in MVTEC_OBJECT_NAMES:
            raise ValueError(f"Unknown MVTec category '{category}'.")
        if category in self._memory_cache:
            return self._memory_cache[category]

        cached = self._load_from_disk(category)
        if cached is None:
            cached = self._create_embeddings(category)
            self._save_to_disk(category, cached)
        self._memory_cache[category] = cached
        return cached

    def _create_embeddings(self, category: str) -> PromptEmbeddings:
        object_name = MVTEC_OBJECT_NAMES[category]

        def build(states: tuple[str, ...]) -> list[str]:
            return [
                template.format(state=state.format(object=object_name))
                for state in states
                for template in TEMPLATES
            ]

        normal_prompts = build(NORMAL_STATES)
        anomalous_prompts = build(ANOMALOUS_STATES)
        normal = functional.normalize(
            self.encoder.encode_text(normal_prompts).mean(dim=0), dim=0
        )
        anomalous = functional.normalize(
            self.encoder.encode_text(anomalous_prompts).mean(dim=0), dim=0
        )
        return PromptEmbeddings(normal=normal, anomalous=anomalous)

    def _cache_paths(self, category: str) -> tuple[Path, Path]:
        stem = f"{category}_{self.cache_key}"
        return self.cache_dir / f"{stem}.pt", self.cache_dir / f"{stem}.json"

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
            normal, anomalous = payload["normal"], payload["anomalous"]
            if normal.shape != (EMBEDDING_DIM,) or anomalous.shape != (EMBEDDING_DIM,):
                return None
            return PromptEmbeddings(normal=normal, anomalous=anomalous)
        except (KeyError, OSError, RuntimeError, json.JSONDecodeError):
            return None

    def _save_to_disk(self, category: str, embeddings: PromptEmbeddings) -> None:
        tensor_path, metadata_path = self._cache_paths(category)
        torch.save({
            "normal": embeddings.normal.cpu(),
            "anomalous": embeddings.anomalous.cpu(),
        }, tensor_path)
        metadata_path.write_text(
            json.dumps({
                "category": category,
                "object_name": MVTEC_OBJECT_NAMES[category],
                "model_name": MODEL_NAME,
                "pretrained_checkpoint": PRETRAINED_CHECKPOINT,
                "open_clip_version": open_clip.__version__,
                "cache_key": self.cache_key,
            }, indent=2),
            encoding="utf-8",
        )
