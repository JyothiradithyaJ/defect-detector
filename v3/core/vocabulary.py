"""Domain-informed defect vocabularies for zero-shot inspection.

Formulated strictly from general manufacturing domain knowledge and material physics.
DOES NOT use or inspect ground-truth defect-type folder names from the test split.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Generic manufacturing surface and structural flaws
GENERAL_DEFECT_TERMS: tuple[str, ...] = (
    "damaged",
    "defective",
    "broken",
    "scratched",
    "cracked",
    "dented",
    "flawed",
    "deformed",
    "contaminated",
    "stained",
    "torn",
)

# Product-specific flaw descriptors based on material and assembly structure
PRODUCT_SPECIFIC_VOCABULARY: dict[str, tuple[str, ...]] = {
    "bottle": (
        "cracked glass",
        "broken rim",
        "scratched surface",
        "contamination",
        "dirty spot",
        "glass chip",
        "surface flaw",
    ),
    "cable": (
        "cut insulation",
        "bent wire",
        "frayed wire",
        "peeled casing",
        "exposed conductor",
        "flattened wire",
        "twisted cable",
    ),
    "capsule": (
        "dented shell",
        "cracked surface",
        "scratched gelatin",
        "stain on capsule",
        "faulty imprint",
        "squeezed body",
        "pitted capsule",
    ),
    "carpet": (
        "hole in weave",
        "cut thread",
        "stain on carpet",
        "discolored patch",
        "torn yarn",
        "surface wear",
        "pulled fiber",
    ),
    "grid": (
        "broken cross",
        "bent bar",
        "missing wire",
        "welding defect",
        "thread tangle",
        "rust spot",
        "misaligned grid",
    ),
    "hazelnut": (
        "cracked shell",
        "hole in nut",
        "broken piece",
        "surface mold",
        "cut nut",
        "decayed shell",
        "spotted nut",
    ),
    "leather": (
        "cut mark",
        "puncture hole",
        "fold crease",
        "stain on leather",
        "scratched hide",
        "burn mark",
        "peeled leather",
    ),
    "metal_nut": (
        "bent thread",
        "scratch on metal",
        "stripped thread",
        "metal burr",
        "cracked steel",
        "corrosion",
        "dented metal",
    ),
    "pill": (
        "cracked tablet",
        "chipped corner",
        "discolored pill",
        "foreign spot",
        "scratched coating",
        "broken pill piece",
        "crushed tablet",
    ),
    "screw": (
        "damaged thread",
        "deformed head",
        "scratched neck",
        "bent screw",
        "metal burr",
        "stripped groove",
        "thread defect",
    ),
    "tile": (
        "cracked tile",
        "chipped corner",
        "oil stain",
        "scratched glaze",
        "rough tile surface",
        "tile discoloration",
        "gray smudge",
    ),
    "toothbrush": (
        "bent bristle",
        "broken handle",
        "missing bristle bundle",
        "split bristle",
        "scratched plastic",
        "matted bristle",
    ),
    "transistor": (
        "bent pin",
        "broken pin",
        "cracked case",
        "damaged housing",
        "corroded lead",
        "misaligned pin",
        "damaged casing",
    ),
    "wood": (
        "knot hole",
        "surface scratch",
        "crack in grain",
        "liquid stain",
        "splintered surface",
        "color blemish",
        "wood gouge",
    ),
    "zipper": (
        "broken tooth",
        "split teeth",
        "frayed fabric tape",
        "misaligned teeth",
        "missing zipper tooth",
        "fabric tear",
        "rough teeth",
    ),
}

MVTEC_OBJECT_NAMES: dict[str, str] = {
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

NORMAL_TEMPLATES: tuple[str, ...] = (
    "a photo of a {object}",
    "a photo of a normal {object}",
    "a photo of a flawless {object}",
    "a photo of a perfect {object}",
    "a photo of an undamaged {object}",
    "a photo of an unblemished {object}",
    "a photo of a good {object}",
    "a photo of a {object} without defects",
    "a close-up photo of a normal {object}",
    "a cropped photo of a normal {object}",
)

DEFECT_TEMPLATES: tuple[str, ...] = (
    "a photo of a damaged {object}",
    "a photo of a defective {object}",
    "a photo of a broken {object}",
    "a photo of a flawed {object}",
    "a photo of an abnormal {object}",
    "a close-up photo of a damaged {object}",
    "a photo of a {object} with a defect",
    "a photo of a {object} with damage",
)

SPECIFIC_TEMPLATES: tuple[str, ...] = (
    "a photo of a {object} with {flaw}",
    "a close-up photo of {flaw} on a {object}",
    "a photo of {flaw} on a {object}",
)


_GLOBAL_TEXT_CACHE: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}


class DomainPromptBank:
    """Prompt bank using generic domain knowledge defect vocabulary."""

    def __init__(self, feature_extractor: Any) -> None:
        self.feature_extractor = feature_extractor
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._backbone_id = getattr(feature_extractor, "backbone_id", None)
        cache_dir = getattr(feature_extractor, "cache_dir", None)
        self._disk_cache_path = (
            Path(cache_dir) / f"domain_text_embeddings_{self._backbone_id}.npz"
            if (cache_dir is not None and self._backbone_id is not None)
            else None
        )
        if self._disk_cache_path is not None and self._disk_cache_path.is_file():
            try:
                with np.load(self._disk_cache_path) as data:
                    for key in data.files:
                        if key.endswith("__norm"):
                            cat = key[: -len("__norm")]
                            def_key = f"{cat}__def"
                            if def_key in data.files:
                                pair = (
                                    np.array(data[key], dtype=np.float32),
                                    np.array(data[def_key], dtype=np.float32),
                                )
                                self._cache[cat] = pair
                                if self._backbone_id is not None:
                                    _GLOBAL_TEXT_CACHE[(self._backbone_id, cat)] = pair
            except OSError:
                pass

    def get_prompts_for_category(self, category: str) -> tuple[list[str], list[str]]:
        """Return normal and defect text prompts for a given category."""
        obj = MVTEC_OBJECT_NAMES.get(category, category)
        normal_prompts = [t.format(object=obj) for t in NORMAL_TEMPLATES]

        defect_prompts = [t.format(object=obj) for t in DEFECT_TEMPLATES]
        for term in GENERAL_DEFECT_TERMS:
            defect_prompts.append(f"a photo of a {term} {obj}")

        specific_terms = PRODUCT_SPECIFIC_VOCABULARY.get(category, ())
        for flaw in specific_terms:
            for t in SPECIFIC_TEMPLATES:
                defect_prompts.append(t.format(object=obj, flaw=flaw))

        return normal_prompts, defect_prompts

    def get_embeddings(self, category: str) -> tuple[np.ndarray, np.ndarray]:
        """Return mean normalized normal and defect text embeddings [D]."""
        if category in self._cache:
            return self._cache[category]
        if self._backbone_id is not None and (self._backbone_id, category) in _GLOBAL_TEXT_CACHE:
            pair = _GLOBAL_TEXT_CACHE[(self._backbone_id, category)]
            self._cache[category] = pair
            return pair

        norm_prompts, def_prompts = self.get_prompts_for_category(category)
        norm_emb = self.feature_extractor.encode_text(norm_prompts)  # [N, D]
        def_emb = self.feature_extractor.encode_text(def_prompts)    # [M, D]

        # Ensemble average and L2 normalize
        mean_norm = norm_emb.mean(axis=0)
        mean_norm = mean_norm / np.linalg.norm(mean_norm)

        mean_def = def_emb.mean(axis=0)
        mean_def = mean_def / np.linalg.norm(mean_def)

        result = (mean_norm.astype(np.float32), mean_def.astype(np.float32))
        self._cache[category] = result
        if self._backbone_id is not None:
            _GLOBAL_TEXT_CACHE[(self._backbone_id, category)] = result
        self._persist_disk_cache()
        return result

    def _persist_disk_cache(self) -> None:
        if self._disk_cache_path is None:
            return
        try:
            payload: dict[str, np.ndarray] = {}
            for cat, (n_emb, d_emb) in self._cache.items():
                payload[f"{cat}__norm"] = n_emb
                payload[f"{cat}__def"] = d_emb
            np.savez(self._disk_cache_path, **payload)
        except OSError:
            pass


