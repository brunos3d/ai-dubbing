"""Path helpers for working directory and project layout."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def project_root() -> Path:
    return PROJECT_ROOT


def working_dir(workdir: Optional[Path] = None) -> Path:
    base = Path(workdir) if workdir is not None else (PROJECT_ROOT / "working")
    base.mkdir(parents=True, exist_ok=True)
    return base


def output_dir(outdir: Optional[Path] = None) -> Path:
    base = Path(outdir) if outdir is not None else (PROJECT_ROOT / "output")
    base.mkdir(parents=True, exist_ok=True)
    return base


def log_dir() -> Path:
    d = PROJECT_ROOT / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def speakers_dir(workdir: Path) -> Path:
    d = workdir / "speakers"
    d.mkdir(parents=True, exist_ok=True)
    return d


def generated_dir(workdir: Path) -> Path:
    d = workdir / "generated_segments"
    d.mkdir(parents=True, exist_ok=True)
    return d


def align_dir(workdir: Path) -> Path:
    d = workdir / "aligned_segments"
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_dir() -> Path:
    d = PROJECT_ROOT / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def hf_cache_dir() -> Path:
    d = models_dir() / "hf_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def env() -> dict:
    """Return environment variables configured for local-first operation."""
    cache = hf_cache_dir()
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache))
    os.environ.setdefault("TORCH_HOME", str(models_dir() / "torch_hub"))
    os.environ.setdefault("XDG_CACHE_HOME", str(models_dir() / "xdg"))
    return os.environ
