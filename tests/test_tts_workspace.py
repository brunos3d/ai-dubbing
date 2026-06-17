"""Workspace integration tests for TTS backend selection.

Verify that the selected backend is persisted in workspace metadata and that
switching it invalidates exactly ``generate..video`` (and nothing upstream),
using the same in-process stub harness as the edit-scenario tests so the real
manifest + invalidation DAG are exercised.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from tests.test_workspace_edit_scenarios import install_stubs


@pytest.fixture
def fixture_audio() -> str:
    return str(Path(__file__).resolve().parent / "fixtures" / "short_sample.wav")


def _prepare(monkeypatch, tmp_path, fixture_audio, backend="omnivoice"):
    from src.workspace.pipeline import WorkspacePipeline

    monkeypatch.setenv("AI_DUBBING_WORKSPACES_ROOT", str(tmp_path))
    record: List[str] = []
    install_stubs(monkeypatch, record)
    wsp = WorkspacePipeline(
        input_path=fixture_audio,
        source_language="en",
        target_language="es",
        tts_backend=backend,
    )
    wid, root = wsp.prepare()
    return record, wid, root


def test_backend_persisted_in_metadata(monkeypatch, tmp_path, fixture_audio):
    _, wid, root = _prepare(monkeypatch, tmp_path, fixture_audio, backend="f5tts")
    meta = json.loads((root / "metadata.json").read_text())
    assert meta["config"]["tts_backend"] == "f5tts"


def test_backend_change_does_not_change_workspace_identity(
    monkeypatch, tmp_path, fixture_audio
):
    """Same media+languages -> same workspace id regardless of backend."""
    _, wid_omni, _ = _prepare(monkeypatch, tmp_path / "a", fixture_audio, "omnivoice")
    _, wid_f5, _ = _prepare(monkeypatch, tmp_path / "b", fixture_audio, "f5tts")
    assert wid_omni == wid_f5


def test_switching_backend_invalidates_generate_downstream(
    monkeypatch, tmp_path, fixture_audio
):
    """Changing the backend re-runs generate..video, not extract..translate."""
    from src.workspace.pipeline import WorkspacePipeline

    # Prepare + first generate with OmniVoice.
    record, wid, root = _prepare(monkeypatch, tmp_path, fixture_audio, "omnivoice")
    assert record == ["extract", "separate", "diarize", "samples", "transcribe", "translate"]

    gen_record: List[str] = []
    install_stubs(monkeypatch, gen_record)
    WorkspacePipeline(
        input_path=fixture_audio, source_language="en", target_language="es",
        tts_backend="omnivoice",
    ).generate(wid)
    assert gen_record == ["generate", "align", "reconstruct", "mix", "video"]

    # Re-generate with the SAME backend: nothing should re-run.
    noop_record: List[str] = []
    install_stubs(monkeypatch, noop_record)
    WorkspacePipeline(
        input_path=fixture_audio, source_language="en", target_language="es",
        tts_backend="omnivoice",
    ).generate(wid)
    assert noop_record == []

    # Now switch to F5-TTS: generate..video re-run; upstream stays cached.
    switch_record: List[str] = []
    install_stubs(monkeypatch, switch_record)
    WorkspacePipeline(
        input_path=fixture_audio, source_language="en", target_language="es",
        tts_backend="f5tts",
    ).generate(wid)
    assert switch_record == ["generate", "align", "reconstruct", "mix", "video"]
    assert "extract" not in switch_record and "translate" not in switch_record

    # Metadata now reflects the switch.
    meta = json.loads((root / "metadata.json").read_text())
    assert meta["config"]["tts_backend"] == "f5tts"
