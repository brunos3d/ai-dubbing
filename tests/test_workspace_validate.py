"""Tests for ``src/workspace/validate.py`` (spec §14).

The validators are imported as a top-level module because ``src/workspace/``
does not yet have an ``__init__.py`` (Task 2 deliverable). When that file
lands, the import style can be switched to ``from src.workspace import
validate`` without changing the test bodies.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "workspace"))

from validate import (  # noqa: E402
    Issue,
    validate_diarization_segments,
    validate_glossary,
    validate_manifest,
    validate_metadata,
    validate_speaker_metadata,
    validate_speaker_sample,
    validate_transcript,
    validate_translation,
)


# ---------------------------------------------------------------------------
# transcript / translation
# ---------------------------------------------------------------------------


def test_validate_transcript_ok() -> None:
    data = {
        "$schema_version": 1,
        "language": "pt",
        "segments": [
            {
                "id": "seg_0001",
                "start": 0.0,
                "end": 1.5,
                "speaker": "speaker_01",
                "text": "Olá mundo",
            }
        ],
    }
    assert validate_transcript(data) == []


def test_validate_transcript_missing_keys() -> None:
    issues = validate_transcript({"segments": []})
    assert any(i.severity == "error" for i in issues), issues


def test_validate_translation_requires_source_text() -> None:
    data = {
        "$schema_version": 1,
        "language": "en",
        "segments": [
            {
                "id": "seg_0001",
                "start": 0.0,
                "end": 1.5,
                "speaker": "speaker_01",
                "text": "Hello world",
            }
        ],
    }
    issues = validate_translation(data)
    assert any(
        i.severity == "warning" and "source_text" in (i.path + i.message)
        for i in issues
    ), issues


# ---------------------------------------------------------------------------
# glossary
# ---------------------------------------------------------------------------


def test_validate_glossary_ok() -> None:
    data = {
        "$schema_version": 1,
        "entries": {"Peter": {"action": "preserve"}},
    }
    assert validate_glossary(data) == []


def test_validate_glossary_rejects_unknown_action() -> None:
    data = {
        "$schema_version": 1,
        "entries": {"Peter": {"action": "delete"}},
    }
    issues = validate_glossary(data)
    assert any(i.severity == "error" for i in issues), issues


# ---------------------------------------------------------------------------
# diarization segments
# ---------------------------------------------------------------------------


def test_validate_diarization_segments_ok() -> None:
    data = {
        "$schema_version": 1,
        "speakers": ["speaker_01", "speaker_02"],
        "segments": [
            {"start": 0.0, "end": 1.5, "speaker": "speaker_01"},
            {"start": 1.5, "end": 3.0, "speaker": "speaker_02"},
        ],
    }
    assert validate_diarization_segments(data) == []


# ---------------------------------------------------------------------------
# bare-list (legacy / runtime) shapes — the stages actually emit these
# ---------------------------------------------------------------------------


def test_validate_transcript_accepts_bare_list() -> None:
    """The transcribe stage emits a bare list of segments (no wrapper dict)."""
    data = [
        {"start": 0.0, "end": 1.5, "speaker": "speaker_01", "text": "Olá"},
        {"start": 1.5, "end": 3.0, "speaker": "speaker_01", "text": "mundo"},
    ]
    assert validate_transcript(data) == []


def test_validate_transcript_bare_list_flags_missing_field() -> None:
    data = [{"start": 0.0, "end": 1.5, "speaker": "speaker_01"}]  # no text
    issues = validate_transcript(data)
    assert any(i.severity == "error" and "text" in i.path for i in issues), issues


def test_validate_translation_bare_list_warns_missing_source_text() -> None:
    data = [{"start": 0.0, "end": 1.5, "speaker": "speaker_01", "text": "hi"}]
    issues = validate_translation(data)
    assert any(
        i.severity == "warning" and "source_text" in i.path for i in issues
    ), issues


def test_validate_diarization_accepts_bare_list() -> None:
    """The diarize stage emits a bare list of {speaker,start,end} dicts."""
    data = [
        {"speaker": "speaker_01", "start": 0.0, "end": 1.5},
        {"speaker": "speaker_02", "start": 1.5, "end": 3.0},
    ]
    assert validate_diarization_segments(data) == []


# ---------------------------------------------------------------------------
# speaker metadata
# ---------------------------------------------------------------------------


def test_validate_speaker_metadata_ok() -> None:
    data = {
        "$schema_version": 1,
        "speaker_id": "speaker_01",
        "voice_profile_hash": "sha256:abc",
        "score": 87.4,
        "snr_db": 28.1,
        "reference_duration_s": 10.0,
        "segments_used": 3,
        "transcript_words": 27,
        "model": "wespeaker-resnet34",
        "extracted_at": "2026-06-13T12:34:56Z",
    }
    assert validate_speaker_metadata(data) == []


def test_validate_speaker_sample_short_audio_is_error(tmp_path: Path) -> None:
    wav = tmp_path / "primary.wav"
    sf.write(str(wav), np.zeros(8000, dtype=np.float32), 16000)
    txt = tmp_path / "primary.txt"
    txt.write_text("hello world")
    issues = validate_speaker_sample(wav, txt)
    assert any(
        i.severity == "error" and "duration" in i.message.lower() for i in issues
    ), issues


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


def test_validate_manifest_ok(tmp_path: Path) -> None:
    data = {
        "schema_version": 1,
        "workspace_id": "peter-20260613-b7c1f4aa",
        "stages": {},
    }
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(data))
    loaded = json.loads(p.read_text())
    assert validate_manifest(loaded) == []


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------


def test_validate_metadata_ok() -> None:
    data = {
        "workspace_schema_version": 1,
        "workspace_id": "peter-20260613-b7c1f4aa",
        "content_hash": "b7c1f4aa",
        "created_at": "2026-06-13T12:34:56Z",
        "updated_at": "2026-06-13T12:40:12Z",
        "workspace_created_with": {
            "pipeline_version": "1.2.3",
            "git_commit": "328942c",
            "python_version": "3.12.4",
        },
        "source": {
            "media_path": "/abs/path/peter-ei-nerd.mp4",
            "media_sha256": "abc",
            "source_language": "pt",
            "target_language": "en",
        },
        "config": {
            "whisper_model": "large-v3",
            "hf_token_available": True,
            "min_speakers": None,
            "max_speakers": 8,
            "no_pyannote": False,
            "target_lufs": -16.0,
            "glossary_path": "/abs/path/entities.json",
        },
        "pipeline_config_hash": "def",
    }
    assert validate_metadata(data) == []
