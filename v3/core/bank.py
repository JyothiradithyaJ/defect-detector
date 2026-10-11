"""Reference memory bank management and DINOv2 feature caching."""

from __future__ import annotations

import hashlib
import os
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance

from v3.core.contracts import Recipe

DEFAULT_DINO_CACHE_DIR = Path("D:/defect_cache/dino_features")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def l2_normalize_np(arr: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """L2-normalize numpy array along `axis`."""
    norms = np.linalg.norm(arr, axis=axis, keepdims=True)
    return (arr / np.maximum(norms, eps)).astype(np.float32)


def estimate_dino_cache_bytes(
    num_images: int,
    short_side: int = 560,
    patch_size: int = 14,
    embed_dim: int = 1024,
) -> int:
    """Estimate disk size in bytes for fp16 DINO features: (grid_h * grid_w + 1) * dim * 2 * images."""
    grid = short_side // patch_size
    tokens_per_image = grid * grid + 1
    return int(tokens_per_image * embed_dim * 2 * num_images)


@dataclass(frozen=True)
class CachedDINOFeatures:
    """Cached DINOv2 visual representation for a single image."""

    patch_tokens: np.ndarray  # [H_p, W_p, D] L2-normalized float32
    cls_token: np.ndarray     # [D] L2-normalized float32


def default_synthetic_feature_fn(
    image: Image.Image,
    short_side: int = 560,
    patch_size: int = 14,
    dim: int = 64,
) -> CachedDINOFeatures:
    """Lightweight deterministic patch descriptor from RGB image (used when no GPU model is loaded in unit tests)."""
    grid = max(4, short_side // patch_size)
    rgb = (
        np.asarray(image.convert("RGB").resize((grid * 4, grid * 4), Image.Resampling.BILINEAR), dtype=np.float32)
        / 255.0
    )
    blocks = rgb.reshape(grid, 4, grid, 4, 3).transpose(0, 2, 1, 3, 4).reshape(grid, grid, 48)
    gx = np.gradient(blocks, axis=1)
    gy = np.gradient(blocks, axis=0)
    raw = np.concatenate([blocks, gx, gy], axis=-1)
    if raw.shape[-1] < dim:
        pad = np.zeros((grid, grid, dim - raw.shape[-1]), dtype=np.float32)
        feat = np.concatenate([raw, pad], axis=-1)
    else:
        feat = raw[..., :dim]
    patch_tokens = l2_normalize_np(feat, axis=-1)
    cls_token = l2_normalize_np(np.mean(patch_tokens, axis=(0, 1)), axis=-1)
    return CachedDINOFeatures(patch_tokens=patch_tokens, cls_token=cls_token)


class DINOFeatureExtractor:
    """DINOv2 ViT-L/14 + registers extractor with fp16 disk cache on D:\\defect_cache\\dino_features."""

    def __init__(
        self,
        backbone_id: str = "dinov2_vitl14_reg",
        short_side: int = 560,
        patch_size: int = 14,
        embed_dim: int = 1024,
        precision: str = "fp16",
        device: str = "cuda",
        cache_dir: Path = DEFAULT_DINO_CACHE_DIR,
        torch_home: Path = Path("D:/ml_cache/torch"),
        feature_fn: Callable[[Image.Image], CachedDINOFeatures] | None = None,
        lazy_load: bool = True,
    ) -> None:
        self.backbone_id = backbone_id
        self.short_side = int(short_side)
        self.patch_size = int(patch_size)
        self.grid_size = self.short_side // self.patch_size
        self.embed_dim = int(embed_dim)
        self.precision = precision
        self.device_name = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.torch_home = Path(torch_home)
        self.feature_fn = feature_fn
        self.model: Any = None

        if not lazy_load and self.feature_fn is None:
            self._ensure_model_loaded()

    def _ensure_model_loaded(self) -> None:
        """Load DINOv2 ViT-L/14 with registers from local cache on D:\\ml_cache\\torch."""
        if self.model is not None or self.feature_fn is not None:
            return

        import torch

        os.environ["TORCH_HOME"] = str(self.torch_home)
        torch.hub.set_dir(str(self.torch_home))
        dev = torch.device(self.device_name if torch.cuda.is_available() else "cpu")

        hub_dir = self.torch_home / "hub" / "facebookresearch_dinov2_main"
        if hub_dir.is_dir():
            model = torch.hub.load(str(hub_dir), self.backbone_id, source="local", pretrained=True)
        else:
            model = torch.hub.load("facebookresearch/dinov2", self.backbone_id, pretrained=True)

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        if self.precision == "fp16" and dev.type == "cuda":
            model = model.half().to(dev)
        else:
            model = model.to(dev)
        self.model = model

    def unload_model(self) -> None:
        """Explicitly release GPU memory per single-model-at-a-time rule."""
        import torch

        if self.model is not None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_cache_key_from_bytes(self, image_bytes: bytes) -> str:
        """Construct deterministic cache key = sha256(backbone:short_side:precision:image_sha256)."""
        img_sha = hashlib.sha256(image_bytes).hexdigest()
        raw = f"{self.backbone_id}:{self.short_side}:{self.precision}:{img_sha}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get_cache_key(self, image_path: Path) -> str:
        """Construct deterministic cache key for an image file on disk."""
        return self.get_cache_key_from_bytes(image_path.read_bytes())

    def extract_pil(self, image: Image.Image) -> CachedDINOFeatures:
        """Extract features from a PIL image in memory."""
        if self.feature_fn is not None:
            return self.feature_fn(image)

        import torch
        from torch.nn import functional

        self._ensure_model_loaded()
        assert self.model is not None
        dev = next(self.model.parameters()).device

        target = self.grid_size * self.patch_size
        rgb = image.convert("RGB").resize((target, target), Image.Resampling.BICUBIC)
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        norm = (arr - IMAGENET_MEAN) / IMAGENET_STD
        batch = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).to(dev)
        if self.precision == "fp16" and dev.type == "cuda":
            batch = batch.half()

        with torch.inference_mode():
            out = self.model.forward_features(batch)
            patch_t = functional.normalize(out["x_norm_patchtokens"].float(), dim=-1)
            cls_t = functional.normalize(out["x_norm_clstoken"].float(), dim=-1)

        g = self.grid_size
        patch_np = patch_t.squeeze(0).reshape(g, g, -1).cpu().numpy().astype(np.float32)
        cls_np = cls_t.squeeze(0).cpu().numpy().astype(np.float32)
        return CachedDINOFeatures(patch_tokens=patch_np, cls_token=cls_np)

    def get_features(self, image_path: Path) -> CachedDINOFeatures:
        """Load cached features from D:\\defect_cache\\dino_features or compute and save as fp16."""
        key = self.get_cache_key(image_path)
        cfile = self.cache_dir / f"{key}.npz"

        if cfile.is_file():
            try:
                with np.load(cfile) as data:
                    return CachedDINOFeatures(
                        patch_tokens=l2_normalize_np(np.array(data["patch_tokens"], dtype=np.float32)),
                        cls_token=l2_normalize_np(np.array(data["cls_token"], dtype=np.float32)),
                    )
            except (OSError, zipfile.BadZipFile, ValueError, KeyError):
                cfile.unlink(missing_ok=True)

        with Image.open(image_path) as img:
            feats = self.extract_pil(img.convert("RGB"))

        save_dtype = np.float16 if self.precision == "fp16" else np.float32
        np.savez(
            cfile,
            patch_tokens=feats.patch_tokens.astype(save_dtype),
            cls_token=feats.cls_token.astype(save_dtype),
        )
        return feats


def augment_reference_image(image: Image.Image, symmetry: str = "none") -> list[Image.Image]:
    """Generate reference views: photometric-only unless recipe.symmetry allows flip/rot."""
    base = image.convert("RGB")
    views: list[Image.Image] = [
        base,
        ImageEnhance.Brightness(base).enhance(1.05),
        ImageEnhance.Contrast(base).enhance(1.05),
    ]
    if symmetry == "flip":
        views.append(base.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
    elif symmetry == "rot":
        views.append(base.rotate(180))
    return views


class ReferenceBank:
    """Reference normal patch and CLS memory bank."""

    def __init__(self, augment_mode: str = "photometric_only") -> None:
        self.augment_mode = augment_mode
        self.patch_bank: np.ndarray | None = None  # [N_views, H_p, W_p, D]
        self.cls_bank: np.ndarray | None = None    # [N_views, D]
        self.mean_cls: np.ndarray | None = None    # [D]
        self.reference_ids: list[str] = []

    def fit_from_features(
        self,
        features_list: list[CachedDINOFeatures],
        reference_ids: list[str] | None = None,
        symmetry: str = "none",
    ) -> None:
        """Populate bank directly from pre-extracted CachedDINOFeatures."""
        if not features_list:
            raise ValueError("Cannot fit ReferenceBank with empty features_list.")

        patches: list[np.ndarray] = []
        cls_tokens: list[np.ndarray] = []

        for feat in features_list:
            p = l2_normalize_np(feat.patch_tokens)
            c = l2_normalize_np(feat.cls_token)
            patches.append(p)
            cls_tokens.append(c)

            if symmetry == "flip":
                patches.append(np.ascontiguousarray(p[:, ::-1, :]))
                cls_tokens.append(c)
            elif symmetry == "rot":
                patches.append(np.ascontiguousarray(np.rot90(p, k=2, axes=(0, 1))))
                cls_tokens.append(c)

        self.patch_bank = np.stack(patches, axis=0).astype(np.float32)
        self.cls_bank = np.stack(cls_tokens, axis=0).astype(np.float32)
        self.mean_cls = l2_normalize_np(np.mean(self.cls_bank, axis=0))
        self.reference_ids = list(reference_ids) if reference_ids is not None else []

    def fit_from_images(
        self,
        normals: list[Image.Image],
        extractor: DINOFeatureExtractor,
        recipe: Recipe | None = None,
        reference_ids: list[str] | None = None,
    ) -> None:
        """Extract features from normal PIL images (with allowed augmentations) and populate bank."""
        if not normals:
            raise ValueError("Cannot fit ReferenceBank with empty normals list.")
        symmetry = recipe.symmetry if recipe is not None else "none"
        extracted: list[CachedDINOFeatures] = []
        for img in normals:
            views = augment_reference_image(img, symmetry=symmetry) if self.augment_mode else [img]
            for v in views:
                extracted.append(extractor.extract_pil(v))
        self.fit_from_features(extracted, reference_ids=reference_ids, symmetry="none")
