"""Tests for timing-aware translation selection in translate_segments.

A stub backend supplies deterministic candidates so no network is touched.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stages.translate import TranslationBackend, translate_segments  # noqa: E402


class _StubBackend(TranslationBackend):
    name = "stub"

    def __init__(self, candidates_by_text):
        self._by_text = candidates_by_text

    def translate(self, text, source, target):
        return self._by_text.get(text, [(text, "stub")])[0][0]

    def translate_candidates(self, text, source, target):
        return self._by_text.get(text, [(text, "stub")])


def test_selects_best_fitting_candidate() -> None:
    # One 3 s slot; offer a short and a fitting candidate.
    segs = [{"speaker": "speaker_01", "start": 0.0, "end": 3.0, "text": "fonte"}]
    backend = _StubBackend({
        "fonte": [
            ("Hi there", "stub:short"),
            ("Hello, how are you doing today my friend", "stub:fit"),
        ]
    })
    out = translate_segments(segs, "pt", "en", backend=backend)
    assert len(out) == 1
    seg = out[0]
    assert "timing" in seg
    assert seg["timing"]["n_candidates"] == 2
    assert seg["source_text"] == "fonte"
    # The longer, better-fitting candidate should be selected for a 3 s slot.
    assert seg["text"] == "Hello, how are you doing today my friend"
    assert seg["translation_backend"] == "stub:fit"


def test_non_speech_segments_passthrough() -> None:
    segs = [{"speaker": "speaker_01", "start": 0.0, "end": 1.0,
             "text": "[laughter]", "is_non_speech": True}]
    backend = _StubBackend({})
    out = translate_segments(segs, "pt", "en", backend=backend)
    assert out[0]["text"] == "[laughter]"
    assert "timing" not in out[0]  # non-speech is not timed/translated


def test_same_language_copies_without_backend() -> None:
    segs = [{"speaker": "speaker_01", "start": 0.0, "end": 2.0, "text": "hello"}]
    out = translate_segments(segs, "en", "en")
    assert out[0]["text"] == "hello"


def test_backend_failure_falls_back_to_original() -> None:
    class _FailBackend(TranslationBackend):
        name = "fail"

        def translate(self, text, source, target):
            raise RuntimeError("boom")

        def translate_candidates(self, text, source, target):
            raise RuntimeError("boom")

    segs = [{"speaker": "speaker_01", "start": 0.0, "end": 2.0, "text": "olá"}]
    out = translate_segments(segs, "pt", "en", backend=_FailBackend())
    # Falls back to the (protected) original; run does not abort.
    assert out[0]["text"] == "olá"
    assert out[0]["translation_backend"] == "passthrough"
