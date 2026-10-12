"""Feature extraction and visual embedding disk cache for CLIP backbones.

Supports backbones:
- clip_b32_224 (ViT-B-32-quickgelu, openai, 224px)
- clip_l14_336 (ViT-L-14-336, openai, 336px)

Provides disk-level visual feature caching under D:/defect_cache/visual_embeddings.
Never downloads weights from the network if the checkpoint already exists on D:/ml_cache.
"""

from __future__ import annotations

import hashlib
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure HuggingFace and Torch cache directories default to D:\ml_cache
if Path("D:/ml_cache/hf").is_dir():
    os.environ["HF_HOME"] = "D:/ml_cache/hf"
if Path("D:/ml_cache/torch").is_dir():
    os.environ["TORCH_HOME"] = "D:/ml_cache/torch"
if Path("D:/ml_cache/tmp").is_dir():
    os.environ["TEMP"] = "D:/ml_cache/tmp"
    os.environ["TMP"] = "D:/ml_cache/tmp"

import numpy as np
import torch
from PIL import Image
from torch.nn import functional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = (
    Path("D:/defect_cache/visual_embeddings")
    if Path("D:/").is_dir()
    else (PROJECT_ROOT / "data" / "cache" / "visual_embeddings")
)

BACKBONE_SPECS: dict[str, dict[str, Any]] = {
    "clip_b32_224": {
        "model_name": "ViT-B-32-quickgelu",
        "pretrained": "openai",
        "hf_repo_dir": "models--timm--vit_base_patch32_clip_224.openai",
        "input_size": 224,
        "embedding_dim": 512,
        "patch_grid": 7,
    },
    "clip_l14_336": {
        "model_name": "ViT-L-14-336",
        "pretrained": "openai",
        "hf_repo_dir": "models--timm--vit_large_patch14_clip_336.openai",
        "input_size": 336,
        "embedding_dim": 768,
        "patch_grid": 24,
    },
}


def resolve_local_checkpoint(backbone_id: str) -> str:
    """Return local checkpoint path on D:/ml_cache if present; never re-download existing weights."""
    spec = BACKBONE_SPECS[backbone_id]
    repo_dir = spec["hf_repo_dir"]
    search_roots = [
        Path(os.environ.get("HF_HOME", "D:/ml_cache/hf")) / "hub" / repo_dir / "snapshots",
        Path(os.environ.get("TORCH_HOME", "D:/ml_cache/torch")) / repo_dir / "snapshots",
        Path("D:/ml_cache/hf/hub") / repo_dir / "snapshots",
        Path("D:/ml_cache/torch") / repo_dir / "snapshots",
    ]
    for root in search_roots:
        if root.is_dir():
            for candidate in root.rglob("open_clip_model.safetensors"):
                if candidate.is_file() and candidate.stat().st_size > 100_000_000:
                    os.environ["HF_HUB_OFFLINE"] = "1"
                    return str(candidate)
            for candidate in root.rglob("open_clip_pytorch_model.bin"):
                if candidate.is_file() and candidate.stat().st_size > 100_000_000:
                    os.environ["HF_HUB_OFFLINE"] = "1"
                    return str(candidate)
    return str(spec["pretrained"])


@dataclass(frozen=True)
class CachedVisualFeatures:
    """Cached visual representations for an image and its horizontal flip."""

    global_embedding: np.ndarray             # [D]
    global_embedding_flipped: np.ndarray     # [D]
    patch_embeddings: np.ndarray             # [H, W, D] (v1 multi-layer intermediate)
    patch_embeddings_flipped: np.ndarray     # [H, W, D]
    final_patch_tokens: np.ndarray           # [H, W, D] (final layer with ln_post + proj)
    final_patch_tokens_flipped: np.ndarray   # [H, W, D]


class CLIPFeatureExtractor:
    """Configurable CLIP backbone wrapper supporting B/32 and L/14 with disk caching."""

    def __init__(
        self,
        backbone_id: str = "clip_b32_224",
        device: str | None = None,
        precision: str = "fp32",
        cache_dir: Path = DEFAULT_CACHE_DIR,
        lazy_load: bool = True,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if backbone_id not in BACKBONE_SPECS:
            raise ValueError(f"Unknown backbone_id '{backbone_id}'. Available: {list(BACKBONE_SPECS.keys())}")

        self.backbone_id = backbone_id
        self.spec = BACKBONE_SPECS[backbone_id]
        self.device = torch.device(device)
        self.precision = precision
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pretrained_source = resolve_local_checkpoint(backbone_id)
        self.model: Any = None
        self.preprocess: Any = None
        self.tokenizer: Any = None
        self._text_only: bool = False

        if not lazy_load:
            self._ensure_model_loaded()

    def _ensure_model_loaded(self) -> None:
        if self.model is not None and not self._text_only:
            return
        import open_clip

        load_device = "cpu" if (self.precision == "fp16" and self.device.type == "cuda") else self.device
        hf_hub_cache = str(Path(os.environ.get("HF_HOME", "D:/ml_cache/hf")) / "hub")

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.spec["model_name"],
            pretrained=self.pretrained_source,
            device=load_device,
            cache_dir=hf_hub_cache,
        )
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        if self.precision == "fp16" and self.device.type == "cuda":
            model = model.half().to(self.device)

        self.model = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(self.spec["model_name"])
        self._text_only = False

    def _ensure_text_model_loaded(self) -> None:
        """Load only the text tower weights from local safetensors to avoid 1.2GB visual tower RAM spike."""
        if self.model is not None:
            return
        import gc

        import open_clip

        ckpt_path = Path(self.pretrained_source)
        if ckpt_path.is_file() and ckpt_path.suffix == ".safetensors":
            from safetensors import safe_open

            model = open_clip.create_model(self.spec["model_name"], pretrained=None, device="cpu")
            model.visual = None
            gc.collect()
            param_map = dict(model.named_parameters())
            buf_map = dict(model.named_buffers())
            with torch.no_grad(), safe_open(str(ckpt_path), framework="pt", device="cpu") as f:
                for k in f.keys():  # noqa: SIM118
                    if k.startswith("visual."):
                        continue
                    if k in param_map:
                        param_map[k].copy_(f.get_tensor(k))
                    elif k in buf_map:
                        buf_map[k].copy_(f.get_tensor(k))
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            if self.device.type == "cuda":
                model = model.to(self.device)
            self.model = model
            self.tokenizer = open_clip.get_tokenizer(self.spec["model_name"])
            self._text_only = True
            return

        self._ensure_model_loaded()

    def compute_image_hash(self, image_path: Path) -> str:
        """Calculate SHA-256 of raw image file."""
        return hashlib.sha256(image_path.read_bytes()).hexdigest()

    def get_cache_key(self, image_path: Path) -> str:
        """Construct unique deterministic cache key."""
        img_hash = self.compute_image_hash(image_path)
        raw_key = f"{self.backbone_id}:{self.spec['input_size']}:{self.precision}:{img_hash}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    @torch.inference_mode()
    def _extract_visual_features_raw(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract global, intermediate patch, and final layer patch features."""
        self._ensure_model_loaded()
        batch = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        if self.precision == "fp16" and self.device.type == "cuda":
            batch = batch.half()

        # 1. Forward intermediate representations
        outputs = self.model.forward_intermediates(
            image=batch,
            image_indices=3,
            normalize=True,
            normalize_intermediates=True,
            image_output_fmt="NLC",
        )
        global_feat = outputs["image_features"]  # [1, D]
        intermediates = outputs["image_intermediates"]  # tuple of [1, num_patches, hidden_dim]

        projection = self.model.visual.proj
        if projection is None:
            raise RuntimeError("Visual projection unavailable.")

        # v1 multi-layer intermediate projection without ln_post
        projected_layers = tuple(
            functional.normalize(layer @ projection, dim=-1)
            for layer in intermediates
        )
        intermediate_patches = projected_layers[-1]  # [1, num_patches, D]

        # 2. Extract final layer patch tokens with ln_post applied
        last_block_tokens = intermediates[-1]
        if hasattr(self.model.visual, "ln_post"):
            ln_tokens = self.model.visual.ln_post(last_block_tokens)
            final_patches = functional.normalize(ln_tokens @ projection, dim=-1)
        else:
            final_patches = intermediate_patches

        return global_feat, intermediate_patches, final_patches

    def get_features(self, image_path: Path) -> CachedVisualFeatures:
        """Retrieve features from disk cache or compute and cache them."""
        cache_key = self.get_cache_key(image_path)
        cache_file = self.cache_dir / f"{cache_key}.npz"

        if cache_file.is_file():
            try:
                with np.load(cache_file) as data:
                    return CachedVisualFeatures(
                        global_embedding=np.array(data["global_embedding"], dtype=np.float32),
                        global_embedding_flipped=np.array(data["global_embedding_flipped"], dtype=np.float32),
                        patch_embeddings=np.array(data["patch_embeddings"], dtype=np.float32),
                        patch_embeddings_flipped=np.array(data["patch_embeddings_flipped"], dtype=np.float32),
                        final_patch_tokens=np.array(data["final_patch_tokens"], dtype=np.float32),
                        final_patch_tokens_flipped=np.array(data["final_patch_tokens_flipped"], dtype=np.float32),
                    )
            except (OSError, zipfile.BadZipFile, ValueError, KeyError):
                try:
                    cache_file.unlink(missing_ok=True)
                except OSError:
                    pass

        with Image.open(image_path) as img:
            original = img.convert("RGB")
            flipped = original.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

            g, p, f = self._extract_visual_features_raw(original)
            gf, pf, ff = self._extract_visual_features_raw(flipped)

        grid = self.spec["patch_grid"]
        dim = self.spec["embedding_dim"]
        save_dtype = np.float16 if self.precision == "fp16" else np.float32

        g_np = g.squeeze(0).float().cpu().numpy().astype(save_dtype)
        gf_np = gf.squeeze(0).float().cpu().numpy().astype(save_dtype)
        p_np = p.squeeze(0).reshape(grid, grid, dim).float().cpu().numpy().astype(save_dtype)
        pf_np = pf.squeeze(0).reshape(grid, grid, dim).float().cpu().numpy().astype(save_dtype)
        f_np = f.squeeze(0).reshape(grid, grid, dim).float().cpu().numpy().astype(save_dtype)
        ff_np = ff.squeeze(0).reshape(grid, grid, dim).float().cpu().numpy().astype(save_dtype)

        np.savez(
            cache_file,
            global_embedding=g_np,
            global_embedding_flipped=gf_np,
            patch_embeddings=p_np,
            patch_embeddings_flipped=pf_np,
            final_patch_tokens=f_np,
            final_patch_tokens_flipped=ff_np,
        )

        return CachedVisualFeatures(
            global_embedding=g_np.astype(np.float32),
            global_embedding_flipped=gf_np.astype(np.float32),
            patch_embeddings=p_np.astype(np.float32),
            patch_embeddings_flipped=pf_np.astype(np.float32),
            final_patch_tokens=f_np.astype(np.float32),
            final_patch_tokens_flipped=ff_np.astype(np.float32),
        )

    @torch.inference_mode()
    def encode_text(self, prompts: list[str]) -> np.ndarray:
        """Encode text prompt strings to normalized numpy vectors [num_prompts, D]."""
        self._ensure_text_model_loaded()
        tokens = self.tokenizer(prompts).to(self.device)
        text_features = self.model.encode_text(tokens, normalize=True)
        return text_features.float().cpu().numpy()


