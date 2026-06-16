"""Tests for per-segment prosody descriptors (spec §5)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.timing.prosody import (  # noqa: E402
    analyze_segment,
    compute_gain,
    compute_speed,
)


def _tone(freq: float, dur: float, sr: int, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype("float32")


def test_louder_segment_has_higher_energy() -> None:
    sr = 16000
    quiet = _tone(150, 1.0, sr, amp=0.05)
    loud = _tone(150, 1.0, sr, amp=0.5)
    dq = analyze_segment(quiet, sr, 0.0, 1.0)
    dl = analyze_segment(loud, sr, 0.0, 1.0)
    assert dl.energy_rms > dq.energy_rms
    assert dl.energy_db > dq.energy_db


def test_pause_ratio_detects_silence() -> None:
    sr = 16000
    speech = _tone(150, 0.5, sr, amp=0.4)
    silence = np.zeros(int(0.5 * sr), dtype="float32")
    clip = np.concatenate([speech, silence])
    d = analyze_segment(clip, sr, 0.0, 1.0)
    # Roughly half the segment is silent.
    assert 0.3 < d.pause_ratio < 0.7


def test_speaking_rate_uses_text_and_voiced_time() -> None:
    sr = 16000
    clip = _tone(150, 2.0, sr, amp=0.4)
    d = analyze_segment(clip, sr, 0.0, 2.0, source_text="one two three four", lang="en")
    assert d.speaking_rate_syl_s is not None
    assert d.speaking_rate_syl_s > 0


def test_signature_is_stable_and_sensitive() -> None:
    sr = 16000
    clip = _tone(150, 1.0, sr, amp=0.4)
    d1 = analyze_segment(clip, sr, 0.0, 1.0, source_text="hello world", lang="en")
    d2 = analyze_segment(clip, sr, 0.0, 1.0, source_text="hello world", lang="en")
    assert d1.signature() == d2.signature()
    louder = analyze_segment(_tone(150, 1.0, sr, amp=0.05), sr, 0.0, 1.0,
                             source_text="hello world", lang="en")
    assert louder.signature() != d1.signature()


def test_compute_speed_band() -> None:
    assert compute_speed(None, 4.0) == 1.0          # missing rate -> neutral
    assert compute_speed(8.0, 4.0) == 1.15          # very fast -> clamped high
    assert compute_speed(1.0, 4.0) == 0.85          # very slow -> clamped low
    assert 0.9 < compute_speed(4.2, 4.0) < 1.1      # near mean -> near 1


def test_compute_gain_band() -> None:
    assert compute_gain(0.0, 0.1) == 1.0            # silent -> neutral
    assert compute_gain(1.0, 0.1) == 1.40           # much louder -> clamped high
    assert compute_gain(0.001, 0.1) == 0.70         # much quieter -> clamped low


def test_empty_segment_is_safe() -> None:
    d = analyze_segment(np.zeros(10, dtype="float32"), 16000, 0.0, 0.0001)
    assert d.energy_rms == 0.0 and d.speaking_rate_syl_s is None
