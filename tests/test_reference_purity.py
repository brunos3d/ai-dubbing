"""Tests for speaker-reference purification (Objectives #1/#2).

The cloning reference must contain the *target voice only*: dense voiced
speech, with silence pockets and low-level ambience/audience bleed removed.
The old reference was ~44% voiced (half silence), which dilutes the speaker
embedding. These cover the pure-Python purification helpers.
"""
from __future__ import annotations

import numpy as np

from src.stages.samples import (
    _voiced_frac,
    trim_to_voiced,
    denoise_spectral_gate,
    purify_reference,
)


def _voiced_tone(sr, dur, f0=140.0, amp=0.3):
    t = np.arange(int(dur * sr)) / sr
    # A few harmonics so it reads as voiced speech, not a pure sine.
    sig = sum(amp / k * np.sin(2 * np.pi * f0 * k * t) for k in (1, 2, 3))
    return sig.astype("float32")


def test_trim_removes_long_silence():
    sr = 16000
    voiced = _voiced_tone(sr, 1.0)
    silence = np.zeros(int(1.5 * sr), dtype="float32")
    sig = np.concatenate([voiced, silence, voiced])
    before = _voiced_frac(sig, sr)
    out = trim_to_voiced(sig, sr)
    after = _voiced_frac(out, sr)
    assert after > before
    assert after > 0.8
    # The long internal silence is gone, so the result is much shorter.
    assert out.shape[0] < sig.shape[0] * 0.8


def test_trim_keeps_voiced_content():
    sr = 16000
    voiced = _voiced_tone(sr, 2.0)
    out = trim_to_voiced(voiced, sr)
    # Already dense -> kept almost entirely.
    assert out.shape[0] > 0.8 * voiced.shape[0]


def test_denoise_lowers_noise_floor_without_killing_voice():
    sr = 16000
    rng = np.random.default_rng(0)
    voiced = _voiced_tone(sr, 1.5)
    noise = 0.02 * rng.standard_normal(voiced.shape[0]).astype("float32")
    noisy = voiced + noise
    out = denoise_spectral_gate(noisy, sr)
    # Voice energy preserved (within a reasonable band).
    v_in = np.sqrt(np.mean(voiced ** 2))
    v_out = np.sqrt(np.mean(out ** 2))
    assert v_out > 0.5 * v_in
    # Residual noise in the gaps is reduced vs input noise level.
    assert out.shape[0] == noisy.shape[0]


def test_purify_raises_voiced_fraction():
    sr = 16000
    rng = np.random.default_rng(1)
    parts = [
        _voiced_tone(sr, 0.8),
        np.zeros(int(0.9 * sr), dtype="float32"),
        _voiced_tone(sr, 0.8),
        0.015 * rng.standard_normal(int(0.6 * sr)).astype("float32"),  # ambience-only
    ]
    sig = np.concatenate(parts)
    pure = purify_reference(sig, sr)
    assert _voiced_frac(pure, sr) > _voiced_frac(sig, sr)
    assert _voiced_frac(pure, sr) > 0.75
    assert pure.shape[0] > int(0.5 * sr)  # didn't nuke everything
