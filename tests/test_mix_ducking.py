"""Tests for acoustic matching: dynamic ducking + room match (spec §6)."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stages.mix import build_mix_filter, estimate_reverb, final_mix  # noqa: E402


def test_ducking_filter_uses_sidechain() -> None:
    fc = build_mix_filter(0.0, -6.0, duck=True)
    assert "sidechaincompress" in fc
    assert "[mix]" in fc
    # Static fallback must NOT use the sidechain.
    static = build_mix_filter(0.0, -6.0, duck=False)
    assert "sidechaincompress" not in static
    assert "amix=inputs=2" in static


def test_reverb_adds_aecho_only_when_wet() -> None:
    dry = build_mix_filter(0.0, -6.0, duck=True, reverb_wet=0.0)
    wet = build_mix_filter(0.0, -6.0, duck=True, reverb_wet=0.2)
    assert "aecho" not in dry
    assert "aecho" in wet


def test_estimate_reverb_dry_vs_wet(tmp_path) -> None:
    sr = 16000
    t = np.arange(int(0.4 * sr)) / sr
    burst = (0.4 * np.sin(2 * np.pi * 160 * t)).astype("float32")
    silence = np.zeros(int(0.4 * sr), dtype="float32")

    # Dry: hard cut to silence after each burst.
    dry = np.concatenate([burst, silence, burst, silence])
    dry_path = tmp_path / "dry.wav"
    sf.write(str(dry_path), dry, sr)

    # Wet: exponentially decaying tail fills the gap after each burst.
    tail_t = np.arange(int(0.4 * sr)) / sr
    tail = (0.4 * np.sin(2 * np.pi * 160 * tail_t) * np.exp(-6 * tail_t)).astype("float32")
    wet = np.concatenate([burst, tail, burst, tail])
    wet_path = tmp_path / "wet.wav"
    sf.write(str(wet_path), wet, sr)

    assert estimate_reverb(wet_path) > estimate_reverb(dry_path)


def test_estimate_reverb_missing_file_is_zero(tmp_path) -> None:
    assert estimate_reverb(tmp_path / "nope.wav") == 0.0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_final_mix_ducking_integration(tmp_path) -> None:
    """End-to-end: the ducking graph actually runs under ffmpeg."""
    sr = 24000
    speech = tmp_path / "speech.wav"
    bg = tmp_path / "bg.wav"
    sf.write(str(speech), (0.3 * np.random.randn(sr * 2)).astype("float32"), sr)
    sf.write(str(bg), (0.2 * np.random.randn(sr * 2)).astype("float32"), sr)
    out = tmp_path / "final.wav"
    info = final_mix(speech, bg, out, duck=True, reverb_wet=0.15)
    assert out.exists()
    assert info["ducking"] is True
