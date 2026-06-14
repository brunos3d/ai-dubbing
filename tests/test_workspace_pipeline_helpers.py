"""Tests for :mod:`src.workspace.pipeline` helpers (Task 10).

The five tests below cover the pure-helper surface of the new
``WorkspacePipeline`` module:

* ``stage_subdir`` — the per-stage workspace-relative subdir lookup.
* ``workspace_layout`` — the canonical list of subdirs a workspace owns.
* ``prepare_stages`` / ``generate_stages`` — the canonical stage ranges for
  the two-phase workflow.

No I/O, no real stages are executed here — these are the contract that the
heavier integration tests in ``tests/test_workspace_e2e.py`` and
``tests/test_workspace_edit_scenarios.py`` rely on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make `src` importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.workspace.pipeline import (  # noqa: E402
    generate_stages,
    prepare_stages,
    stage_subdir,
    workspace_layout,
)


def test_stage_subdir():
    assert stage_subdir("extract") == "media"
    assert stage_subdir("diarize") == "diarization"
    assert stage_subdir("transcribe") == "transcription"
    assert stage_subdir("translate") == "translation"


def test_stage_subdir_root_for_synthesis_stages():
    for s in ("generate", "align", "reconstruct", "mix", "video"):
        assert stage_subdir(s) is None, f"{s} should be at workspace root"


def test_workspace_layout_lists_all_dirs():
    layout = workspace_layout()
    for d in (
        "media",
        "diarization",
        "transcription",
        "translation",
        "speakers",
        "output",
        "source",
        "logs",
    ):
        assert d in layout, f"missing {d!r} in workspace_layout()"


def test_prepare_stages_stops_at_translate():
    assert prepare_stages() == [
        "extract",
        "separate",
        "diarize",
        "samples",
        "transcribe",
        "translate",
    ]


def test_generate_stages_starts_at_generate():
    assert generate_stages() == [
        "generate",
        "align",
        "reconstruct",
        "mix",
        "video",
    ]
