"""Centralized configuration registry for pipeline parameters."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..utils.logging import get_logger
from ..utils.paths import project_root

LOG = get_logger("ai-dubbing.config")


def config_root() -> Path:
    """Return the configuration directory."""
    return project_root() / "config"


def pipeline_defaults_path() -> Path:
    """Return the path to the pipeline defaults file."""
    return config_root() / "pipeline.defaults.json"


def load_pipeline_defaults() -> Dict[str, Any]:
    """Load the pipeline defaults from the registry.

    Returns an empty dict if the file is missing or invalid.
    """
    path = pipeline_defaults_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        LOG.warning("Failed to load pipeline defaults from %s: %s", path, exc)
        return {}


def save_pipeline_defaults(config: Dict[str, Any]) -> None:
    """Save the pipeline defaults to the registry."""
    path = pipeline_defaults_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
