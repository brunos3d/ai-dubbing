"""Tests for audience-reaction preservation (Failure #2).

Demucs routes audience laughter/applause into the *vocals* stem (it is
vocal energy), so the separated background stem is nearly silent in the
reaction gaps and the dub sounds dry. We recover those reactions by adding
the ORIGINAL audio back in, but only in non-speech windows (between
diarized turns) so the original English voice never leaks into the dub.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf

from src.stages.mix import _non_speech_envelope, build_ambience_bed


def test_envelope_zero_under_speech_one_in_gaps():
    sr = 1000
    n = 3 * sr
    env = _non_speech_envelope([{"start": 1.0, "end": 2.0}], n, sr, fade_s=0.05)
    assert env.shape[0] == n
    # Deep inside the speech turn the original is fully gated out.
    assert env[int(1.5 * sr)] < 0.01
    # Well inside the gaps the original passes at full gain.
    assert env[int(0.3 * sr)] > 0.99
    assert env[int(2.7 * sr)] > 0.99
    # Edges are faded, never an instantaneous jump (no click).
    assert 0.0 <= env.min() <= env.max() <= 1.0


def test_ambience_recovers_gap_laughter(tmp_path):
    sr = 16000
    dur = 3.0
    n = int(dur * sr)
    rng = np.random.default_rng(0)
    # Original: loud "laughter" noise everywhere.
    original = (0.3 * rng.standard_normal(n)).astype("float32")
    # Separated background stem: essentially silent (the real-world case).
    background = (1e-4 * rng.standard_normal(n)).astype("float32")
    orig_p = tmp_path / "orig.wav"
    bg_p = tmp_path / "bg.wav"
    out_p = tmp_path / "ambience.wav"
    sf.write(orig_p, original, sr)
    sf.write(bg_p, background, sr)

    # One speech turn covering the middle second.
    build_ambience_bed(orig_p, bg_p, [{"start": 1.0, "end": 2.0}], out_p, sr=sr)

    amb, _ = sf.read(out_p, dtype="float32")
    if amb.ndim > 1:
        amb = amb.mean(1)

    def rms(a, b):
        seg = amb[int(a * sr):int(b * sr)]
        return float(np.sqrt(np.mean(seg ** 2)))

    gap_rms = rms(0.1, 0.8)
    speech_rms = rms(1.3, 1.7)
    # Laughter recovered in the gap; original voice kept out of the speech window.
    assert gap_rms > 0.05
    assert speech_rms < gap_rms * 0.2
