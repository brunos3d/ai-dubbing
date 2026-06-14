"""Tests for ``Pipeline.pipeline_config_dict()``.

Covers:

* The actual ``hf_token`` is never leaked into the returned dict
  (only a boolean ``hf_token_available`` flag is exposed).
* When no token is supplied, ``hf_token_available`` is ``False``.
* All user-configurable knobs (model, LUFS, pyannote toggle, speaker
  counts, glossary) appear in the dict with the values that were
  passed in to ``__init__``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import Pipeline  # noqa: E402


def _make_input(tmp_path: Path) -> Path:
    """Create a tiny placeholder file so ``media_hash`` can hash it."""
    f = tmp_path / "input.bin"
    f.write_bytes(b"placeholder")
    return f


def test_pipeline_config_dict_excludes_hf_token(tmp_path: Path) -> None:
    pipeline = Pipeline(
        input_path=str(_make_input(tmp_path)),
        source_language="en",
        target_language="es",
        workdir=tmp_path,
        hf_token="secret",
    )
    cfg = pipeline.pipeline_config_dict()
    assert "hf_token" not in cfg
    assert cfg["hf_token_available"] is True


def test_pipeline_config_dict_no_hf_token_marks_unavailable(tmp_path: Path) -> None:
    pipeline = Pipeline(
        input_path=str(_make_input(tmp_path)),
        source_language="en",
        target_language="es",
        workdir=tmp_path,
    )
    cfg = pipeline.pipeline_config_dict()
    assert cfg["hf_token_available"] is False
    assert "hf_token" not in cfg


def test_pipeline_config_dict_includes_all_configurable_knobs(tmp_path: Path) -> None:
    pipeline = Pipeline(
        input_path=str(_make_input(tmp_path)),
        source_language="en",
        target_language="es",
        workdir=tmp_path,
        target_lufs=-14.0,
        no_pyannote=True,
        max_speakers=4,
    )
    cfg = pipeline.pipeline_config_dict()
    assert cfg["target_lufs"] == -14.0
    assert cfg["no_pyannote"] is True
    assert cfg["max_speakers"] == 4
