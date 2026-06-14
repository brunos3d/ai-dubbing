"""Tests for src/workspace/manifest.py (Task 4)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.workspace.manifest import (  # noqa: E402
    ArtifactRef,
    Manifest,
    ManifestError,
    StageRecord,
)


def _artifact(path: str, content: bytes = b"x") -> ArtifactRef:
    import hashlib

    return ArtifactRef(
        path=path, sha256=hashlib.sha256(content).hexdigest(), size_bytes=len(content)
    )


def test_minimal_manifest_round_trip(tmp_path: Path) -> None:
    m = Manifest.create(workspace_id="ws-1", pipeline_version="0.1.0", git_commit="abc")
    m.add_stage(
        "extract",
        StageRecord(
            name="extract",
            status="done",
            inputs=[],
            outputs=[_artifact("media/original_audio.wav", b"hello")],
        ),
    )
    m.add_editable_path("transcription/transcript.json")
    m.add_derived_path("generated_segments/manifest.json")

    out = tmp_path / "manifest.json"
    m.save(out)
    loaded = Manifest.load(out)
    assert loaded is not None
    assert loaded.workspace_id == "ws-1"
    assert loaded.pipeline_version == "0.1.0"
    assert loaded.git_commit == "abc"
    assert "transcription/transcript.json" in loaded.editable_paths
    assert "generated_segments/manifest.json" in loaded.derived_paths
    assert "extract" in loaded.stages
    out_art = loaded.stages["extract"].outputs[0]
    assert out_art.path == "media/original_audio.wav"
    assert out_art.sha256 == _artifact("media/original_audio.wav", b"hello").sha256


def test_load_missing_returns_none(tmp_path: Path) -> None:
    assert Manifest.load(tmp_path / "nope.json") is None


def test_load_broken_json_raises(tmp_path: Path) -> None:
    p = tmp_path / "manifest.json"
    p.write_text("not json")
    with pytest.raises(ManifestError):
        Manifest.load(p)


def test_save_uses_atomic_replace(tmp_path: Path) -> None:
    m = Manifest.create(workspace_id="ws-1", pipeline_version="0.1.0", git_commit="abc")
    p = tmp_path / "manifest.json"
    m.save(p)
    leftover = list(tmp_path.iterdir())
    assert leftover == [p]
    tmp_like = [x for x in tmp_path.iterdir() if x.suffix == ".tmp" or ".tmp" in x.name]
    assert tmp_like == []


def test_find_producer_finds_stage_that_emitted_path(tmp_path: Path) -> None:
    m = Manifest.create(workspace_id="ws-1", pipeline_version="0.1.0", git_commit="abc")
    m.add_stage(
        "extract",
        StageRecord(
            name="extract",
            status="done",
            outputs=[_artifact("media/original_audio.wav", b"hi")],
        ),
    )
    assert m.find_producer("media/original_audio.wav") == "extract"


def test_find_producer_returns_none_when_not_found(tmp_path: Path) -> None:
    m = Manifest.create(workspace_id="ws-1", pipeline_version="0.1.0", git_commit="abc")
    m.add_stage(
        "extract",
        StageRecord(
            name="extract",
            status="done",
            outputs=[_artifact("media/original_audio.wav", b"hi")],
        ),
    )
    assert m.find_producer("media/speech.wav") is None


def test_get_input_returns_record(tmp_path: Path) -> None:
    m = Manifest.create(workspace_id="ws-1", pipeline_version="0.1.0", git_commit="abc")
    m.add_stage(
        "separate",
        StageRecord(
            name="separate",
            status="done",
            inputs=[_artifact("media/original_audio.wav", b"hi")],
            outputs=[],
        ),
    )
    ref = m.get_input("separate", "media/original_audio.wav")
    assert ref is not None
    assert ref.path == "media/original_audio.wav"
    # separate has no inputs to look up for "extract"
    assert m.get_input("separate", "media/nonexistent.wav") is None
