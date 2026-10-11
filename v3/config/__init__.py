"""Configuration loader, environment setup, and logging utilities for v3."""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "default.yaml"


@dataclass(frozen=True)
class V3Config:
    """Resolved v3 configuration and D-drive storage paths."""

    raw: dict[str, Any]
    config_hash: str
    data_root: Path
    manifest_dir: Path
    cache_dir: Path
    results_dir: Path
    logs_dir: Path
    repo_results_dir: Path
    decisions_md: Path
    phase1_baseline_json: Path
    hf_home: Path
    torch_home: Path
    temp_dir: Path


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def load_v3_config(config_path: Path = DEFAULT_CONFIG_PATH) -> V3Config:
    """Load default.yaml, configure D-drive environment variables, and return V3Config."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    raw_bytes = config_path.read_bytes()
    config_hash = hashlib.sha256(raw_bytes).hexdigest()[:12]
    raw = yaml.safe_load(raw_bytes.decode("utf-8")) or {}
    paths = raw.get("paths", {})

    hf_home = _resolve_path(paths.get("hf_home", "D:/ml_cache/hf"))
    torch_home = _resolve_path(paths.get("torch_home", "D:/ml_cache/torch"))
    temp_dir = _resolve_path(paths.get("temp_dir", "D:/ml_cache/tmp"))

    # Set environment variables before any ML libraries initialize caches
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["TORCH_HOME"] = str(torch_home)
    os.environ["TEMP"] = str(temp_dir)
    os.environ["TMP"] = str(temp_dir)

    data_root_cfg = _resolve_path(paths.get("data_root", "D:/data/mvtec_ad"))
    if not data_root_cfg.is_dir():
        fallback_data = PROJECT_ROOT / "data" / "mvtec_ad"
        if fallback_data.is_dir():
            data_root_cfg = fallback_data

    cfg = V3Config(
        raw=raw,
        config_hash=config_hash,
        data_root=data_root_cfg,
        manifest_dir=_resolve_path(paths.get("manifest_dir", "data/manifests")),
        cache_dir=_resolve_path(paths.get("cache_dir", "D:/defect_cache/visual_embeddings")),
        results_dir=_resolve_path(paths.get("results_dir", "D:/defect_cache/results")),
        logs_dir=_resolve_path(paths.get("logs_dir", "D:/defect_cache/logs")),
        repo_results_dir=_resolve_path(paths.get("repo_results_dir", "v3/docs/results")),
        decisions_md=_resolve_path(paths.get("decisions_md", "v3/docs/decisions.md")),
        phase1_baseline_json=_resolve_path(
            paths.get("phase1_baseline_json", "v3/docs/results/phase_1_logs/v3-category_results.json")
        ),
        hf_home=hf_home,
        torch_home=torch_home,
        temp_dir=temp_dir,
    )

    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    cfg.repo_results_dir.mkdir(parents=True, exist_ok=True)

    return cfg


class StepLogger:
    """Tee logger writing simultaneously to stdout and D:\\defect_cache\\logs\\<name>.log."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("", encoding="utf-8")

    def info(self, msg: str) -> None:
        print(msg, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def error(self, msg: str) -> None:
        err_line = f"[ERROR] {msg}"
        print(err_line, file=sys.stderr, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(err_line + "\n")

    def close(self) -> None:
        pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()


def log_decision(cfg: V3Config, step_name: str, status: str, seconds: float, detail: str = "") -> None:
    """Append one line to decisions.md: step, status, seconds taken."""
    line = f"- **{step_name}**: status={status}, elapsed={seconds:.2f}s"
    if detail:
        line += f" ({detail})"
    line += "\n"
    cfg.decisions_md.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.decisions_md, "a", encoding="utf-8") as f:
        f.write(line)
