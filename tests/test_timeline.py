"""Tests for Timeline v1 (spec §3)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.workspace.timeline import (  # noqa: E402
    Timeline,
    TimelineSegment,
    build_from_workspace,
    refresh_timeline,
    segment_render_key,
)


def _make_workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    (root / "translation").mkdir(parents=True)
    (root / "transcription").mkdir(parents=True)
    (root / "metadata.json").write_text(json.dumps({
        "workspace_id": "demo-20260616-abcdef01",
        "source": {"source_language": "pt", "target_language": "en"},
    }))
    (root / "transcription" / "transcript.json").write_text(json.dumps([
        {"speaker": "speaker_01", "start": 0.0, "end": 3.0, "text": "Olá mundo"},
    ]))
    (root / "translation" / "translated_transcript.json").write_text(json.dumps([
        {
            "speaker": "speaker_01", "start": 0.0, "end": 3.0,
            "source_text": "Olá mundo", "text": "Hello world",
            "timing": {"slot_duration_s": 3.0, "estimated_duration_s": 2.9,
                       "speaking_rate_factor": 0.97, "score": 0.9, "n_candidates": 2},
        },
    ]))
    return root


def test_build_from_workspace_basic(tmp_path) -> None:
    root = _make_workspace(tmp_path)
    tl = build_from_workspace(root)
    assert tl.language_pair == "pt->en"
    assert len(tl.segments) == 1
    seg = tl.segments[0]
    assert seg.id == "seg_0001"
    assert seg.source_text == "Olá mundo"
    assert seg.target_text == "Hello world"
    assert seg.timing["score"] == 0.9
    assert seg.render_key  # non-empty stable hash
    assert seg.review["low_confidence"] is False


def test_low_score_flags_review(tmp_path) -> None:
    root = _make_workspace(tmp_path)
    data = json.loads((root / "translation" / "translated_transcript.json").read_text())
    data[0]["timing"]["score"] = 0.3
    (root / "translation" / "translated_transcript.json").write_text(json.dumps(data))
    tl = build_from_workspace(root)
    assert tl.segments[0].review["low_confidence"] is True


def test_save_load_roundtrip(tmp_path) -> None:
    root = _make_workspace(tmp_path)
    tl = refresh_timeline(root)
    assert (root / "timeline.json").exists()
    loaded = Timeline.load(root / "timeline.json")
    assert loaded is not None
    assert loaded.to_dict() == tl.to_dict()


def test_absent_timeline_is_tolerated(tmp_path) -> None:
    assert Timeline.load(tmp_path / "nope.json") is None


def test_render_key_sensitivity() -> None:
    base = dict(target_text="Hello", speaker="speaker_01", target_language="en",
                slot_duration_s=3.0)
    k0 = segment_render_key(**base)
    assert k0 == segment_render_key(**base)  # stable
    assert segment_render_key(**{**base, "target_text": "Hi"}) != k0
    assert segment_render_key(**{**base, "speaker": "speaker_02"}) != k0
    assert segment_render_key(**{**base, "slot_duration_s": 4.0}) != k0
    assert segment_render_key(**{**base, "prosody_signature": "x"}) != k0


def test_summary_counts(tmp_path) -> None:
    root = _make_workspace(tmp_path)
    tl = build_from_workspace(root)
    s = tl.summary()
    assert s["segments"] == 1
    assert s["mean_timing_score"] == 0.9


def test_prepared_only_workspace_has_no_target(tmp_path) -> None:
    """A workspace with only a transcript (no translation) still builds."""
    root = tmp_path / "ws2"
    (root / "transcription").mkdir(parents=True)
    (root / "metadata.json").write_text(json.dumps({
        "workspace_id": "x-20260616-00000000",
        "source": {"source_language": "pt", "target_language": "en"},
    }))
    (root / "transcription" / "transcript.json").write_text(json.dumps([
        {"speaker": "speaker_01", "start": 0.0, "end": 2.0, "text": "oi"},
    ]))
    tl = build_from_workspace(root)
    assert len(tl.segments) == 1
    assert tl.segments[0].source_text == "oi"
    assert tl.segments[0].target_text == ""
