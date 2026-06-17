"""Tests for word-level speaker assignment (Failure #1: speaker confusion).

The transcribe stage used to assign ONE diarization speaker per whisper
segment by max time-overlap.  Whisper segments routinely straddle a
speaker turn boundary, so the minority speaker's words were mis-attributed
to the majority speaker and rendered in the wrong cloned voice.

The fix assigns a speaker to every *word* and splits the utterance stream
wherever the speaker changes.
"""
from __future__ import annotations

from src.stages.transcribe import (
    _assign_word_speaker,
    _segment_words_by_speaker,
)


# Diarization mirrors the real Elon/Colbert workspace around 28-38s.
DIAR = [
    {"speaker": "speaker_01", "start": 28.212, "end": 33.511},
    {"speaker": "speaker_03", "start": 35.941, "end": 37.493},
]


def _w(word, start, end):
    return {"word": word, "start": start, "end": end, "logprob": 0.9}


def test_word_speaker_uses_containing_segment():
    # A word inside Colbert's turn -> speaker_01.
    assert _assign_word_speaker(32.2, 32.6, DIAR) == "speaker_01"
    # A word inside Elon's turn -> speaker_03.
    assert _assign_word_speaker(36.0, 36.4, DIAR) == "speaker_03"


def test_word_in_gap_falls_back_to_nearest_speaker():
    # 34.5s lies in the silent gap; nearest turn is speaker_01 (ends 33.5)
    # vs speaker_03 (starts 35.9). 34.5 is closer to speaker_01's end.
    assert _assign_word_speaker(34.4, 34.6, DIAR) == "speaker_01"
    # 35.0s is closer to speaker_03's start.
    assert _assign_word_speaker(35.0, 35.2, DIAR) == "speaker_03"


def test_segment_splits_at_speaker_change():
    """A single whisper block spanning two speakers must split in two."""
    words = [
        _w("You", 32.1, 32.3),
        _w("have", 32.3, 32.5),
        _w("to", 32.5, 32.6),
        _w("choose", 32.6, 32.9),
        _w("one.", 32.9, 33.3),
        _w("I'm", 35.5, 35.7),
        _w("trying", 35.7, 36.0),
        _w("to", 36.0, 36.1),
        _w("do", 36.1, 36.3),
        _w("useful", 36.3, 36.6),
        _w("things", 36.6, 37.08),
    ]
    segs = _segment_words_by_speaker(words, DIAR)
    assert len(segs) == 2
    assert segs[0]["speaker"] == "speaker_01"
    assert "choose one" in segs[0]["text"].lower()
    assert segs[1]["speaker"] == "speaker_03"
    assert "useful things" in segs[1]["text"].lower()


def test_spurious_short_flips_are_smoothed():
    """Noisy 1-word speaker flips in crosstalk collapse into the real turn.

    A long Colbert run with a single mis-diarized word in the middle must
    not shatter into three segments; the stray word is absorbed.
    """
    diar = [
        {"speaker": "speaker_01", "start": 0.0, "end": 5.0},
        {"speaker": "speaker_03", "start": 2.1, "end": 2.4},  # stray blip
    ]
    words = [
        _w("we", 0.2, 0.5),
        _w("could", 0.5, 0.9),
        _w("turn", 0.9, 1.3),
        _w("Mars", 1.3, 1.8),
        _w("into", 2.15, 2.35),  # lands on the blip -> speaker_03
        _w("an", 3.0, 3.3),
        _w("Earth", 3.3, 3.8),
        _w("planet", 3.8, 4.5),
    ]
    segs = _segment_words_by_speaker(words, diar)
    # The single stray word is absorbed; we get one coherent speaker_01 turn.
    assert len(segs) == 1
    assert segs[0]["speaker"] == "speaker_01"
    assert "into" in segs[0]["text"]


def test_pure_single_speaker_stays_single():
    words = [_w("Give", 150.0, 150.3), _w("me", 150.3, 150.5), _w("the", 150.5, 150.7)]
    diar = [{"speaker": "speaker_01", "start": 150.0, "end": 151.0}]
    segs = _segment_words_by_speaker(words, diar)
    assert len(segs) == 1
    assert segs[0]["speaker"] == "speaker_01"
