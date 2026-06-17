"""Tests for transcript-quality fixes (Failure #4: ASR artifacts).

Two concrete defects in the workspace transcript:

* corrupted tokens carrying the Unicode replacement char U+FFFD
  ("It's very in Auf �ative.");
* stutter runs from the recogniser ("initially initially initially").

Plus the low-confidence re-verification gate never fired because the
stored ``avg_logprob`` is actually a probability in [0, 1] while the
threshold was a log-prob (-0.8), so ``avg_logprob >= -0.8`` was always
true.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from src.stages.transcribe import clean_transcript_text, _reverify_low_confidence


def test_replacement_char_token_removed():
    bad = "It's very in Auf �ative. Oh, that's true"
    out = clean_transcript_text(bad)
    assert "�" not in out
    assert "Auf" not in out  # the corrupt token is dropped, not left dangling
    assert "Oh, that's true" in out
    assert "  " not in out  # no double spaces left behind


def test_stutter_run_collapsed():
    out = clean_transcript_text("you're just initially initially initially")
    assert out.lower().count("initially") == 1
    assert out.endswith("initially")


def test_clean_preserves_normal_text():
    good = "The fast way is drop thermonuclear weapons over the poles."
    assert clean_transcript_text(good) == good


def test_double_word_not_collapsed():
    # Only runs of >= 3 are stutters; legitimate doubling ("had had") stays.
    assert clean_transcript_text("that that") == "that that"


def test_reverify_skips_confident_segment():
    """A confident segment (prob 0.95) must not be re-verified."""
    model = MagicMock()
    seg = {"text": "hello", "start": 0.0, "end": 1.0, "avg_logprob": 0.95}
    out = _reverify_low_confidence(
        Path("x.wav"), seg, "tiny", "en", "cpu", "int8", model=model
    )
    model.transcribe.assert_not_called()
    assert out is seg


def test_reverify_triggers_on_low_confidence():
    """A low-confidence segment (prob 0.40) must be re-verified."""
    model = MagicMock()
    model.transcribe.return_value = ([], None)
    seg = {"text": "grbl", "start": 0.0, "end": 1.0, "avg_logprob": 0.40}
    _reverify_low_confidence(
        Path("x.wav"), seg, "tiny", "en", "cpu", "int8", model=model
    )
    model.transcribe.assert_called_once()
