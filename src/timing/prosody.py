"""Per-segment prosody descriptors from the source audio (spec §5).

The synthesizer otherwise reads the translation *flatly*. This module pulls
the cheap, robust prosodic cues out of the already-available ``speech.wav`` so
the pipeline can transfer *delivery*, not just words:

* **speaking rate** (source syllables / voiced time) → drives the TTS
  ``speed`` so a fast or slow line stays fast or slow in the dub;
* **relative energy** (RMS vs the speaker's mean) → drives a per-segment gain
  in reconstruction so emphasis/de-emphasis across a turn is preserved;
* **pitch mean / range** and **pause ratio** → recorded on the Timeline as
  editable, inspectable descriptors (and folded into the render key so a
  prosody change re-renders the segment).

Everything is numpy + (optional) librosa — CPU-only, no torch, no network.
Pitch extraction is guarded so it never dominates runtime on long clips.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .duration import estimate_syllables

# Bands for mapping relative prosody to synthesis controls.
SPEED_LOW, SPEED_HIGH = 0.85, 1.15
GAIN_LOW, GAIN_HIGH = 0.70, 1.40

# Pitch extraction is skipped above this segment length (seconds) to keep
# long-clip analysis cheap; energy/rate/pause are always computed.
_MAX_PITCH_SECONDS = 25.0
_FRAME = 0.025  # 25 ms analysis frame for energy/voicing
_HOP = 0.010    # 10 ms hop


@dataclass
class ProsodyDescriptor:
    """Acoustic delivery descriptors for one segment."""

    energy_rms: float = 0.0
    energy_db: float = -120.0
    voiced_ratio: float = 0.0
    pause_ratio: float = 0.0
    speaking_rate_syl_s: Optional[float] = None
    pitch_mean_hz: Optional[float] = None
    pitch_range_semitones: Optional[float] = None

    def signature(self) -> str:
        """Compact, stable string folded into the segment render key.

        Rounded so that imperceptible numeric jitter does not invalidate a
        render, but a real delivery change does.
        """
        parts = [
            f"e{round(self.energy_db, 1)}",
            f"v{round(self.voiced_ratio, 2)}",
            f"r{round(self.speaking_rate_syl_s, 2) if self.speaking_rate_syl_s else 0}",
            f"p{round(self.pitch_mean_hz, 0) if self.pitch_mean_hz else 0}",
        ]
        return "|".join(parts)

    def to_dict(self) -> dict:
        return {
            "energy_rms": round(self.energy_rms, 5),
            "energy_db": round(self.energy_db, 2),
            "voiced_ratio": round(self.voiced_ratio, 3),
            "pause_ratio": round(self.pause_ratio, 3),
            "speaking_rate_syl_s": (
                round(self.speaking_rate_syl_s, 3)
                if self.speaking_rate_syl_s is not None else None
            ),
            "pitch_mean_hz": (
                round(self.pitch_mean_hz, 1) if self.pitch_mean_hz is not None else None
            ),
            "pitch_range_semitones": (
                round(self.pitch_range_semitones, 2)
                if self.pitch_range_semitones is not None else None
            ),
            "signature": self.signature(),
        }


def _db(x: float) -> float:
    return 20.0 * math.log10(x) if x > 1e-9 else -120.0


def _frame_rms(clip: np.ndarray, sr: int) -> np.ndarray:
    win = max(1, int(_FRAME * sr))
    hop = max(1, int(_HOP * sr))
    if clip.shape[0] < win:
        return np.array([float(np.sqrt(np.mean(clip ** 2)))]) if clip.size else np.array([0.0])
    n = 1 + (clip.shape[0] - win) // hop
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        seg = clip[i * hop: i * hop + win]
        out[i] = float(np.sqrt(np.mean(seg ** 2)))
    return out


def _pitch(clip: np.ndarray, sr: int) -> tuple[Optional[float], Optional[float]]:
    """Return (median pitch Hz, p10..p90 range in semitones) or (None, None)."""
    dur = clip.shape[0] / sr if sr else 0.0
    if dur <= 0.05 or dur > _MAX_PITCH_SECONDS:
        return None, None
    try:
        import librosa  # type: ignore

        f0 = librosa.yin(
            clip.astype("float32"),
            fmin=70.0, fmax=400.0, sr=sr,
            frame_length=max(256, int(0.04 * sr)),
        )
        f0 = f0[np.isfinite(f0)]
        f0 = f0[(f0 > 70.0) & (f0 < 400.0)]
        if f0.size < 3:
            return None, None
        median = float(np.median(f0))
        lo, hi = np.percentile(f0, [10, 90])
        rng = float(12.0 * np.log2(hi / lo)) if lo > 0 else None
        return median, rng
    except Exception:  # noqa: BLE001 - librosa missing or numerical edge
        return None, None


def analyze_segment(
    mono: np.ndarray,
    sr: int,
    start: float,
    end: float,
    *,
    source_text: Optional[str] = None,
    lang: Optional[str] = None,
    silence_rms_frac: float = 0.2,
) -> ProsodyDescriptor:
    """Compute prosody descriptors for ``mono[start:end]``.

    ``source_text``/``lang`` enable the speaking-rate estimate (syllables ÷
    voiced time). Without them, rate is left ``None`` and only the acoustic
    descriptors are filled.
    """
    s0 = max(0, int(start * sr))
    s1 = min(mono.shape[0], int(end * sr))
    if s1 - s0 < int(0.02 * sr):
        return ProsodyDescriptor()
    clip = mono[s0:s1].astype("float64")

    rms = float(np.sqrt(np.mean(clip ** 2)))
    frames = _frame_rms(clip, sr)
    peak = float(frames.max()) if frames.size else 0.0
    thresh = silence_rms_frac * peak if peak > 0 else 0.0
    voiced = frames > thresh if frames.size else np.array([False])
    voiced_ratio = float(np.mean(voiced)) if voiced.size else 0.0
    pause_ratio = 1.0 - voiced_ratio

    total_dur = (s1 - s0) / sr
    voiced_time = max(1e-3, voiced_ratio * total_dur)
    rate: Optional[float] = None
    if source_text:
        syl = estimate_syllables(source_text, lang)
        if syl > 0:
            rate = syl / voiced_time

    pitch_mean, pitch_range = _pitch(clip, sr)

    return ProsodyDescriptor(
        energy_rms=rms,
        energy_db=_db(rms),
        voiced_ratio=voiced_ratio,
        pause_ratio=pause_ratio,
        speaking_rate_syl_s=rate,
        pitch_mean_hz=pitch_mean,
        pitch_range_semitones=pitch_range,
    )


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_speed(rate: Optional[float], mean_rate: Optional[float]) -> float:
    """Map a segment's relative speaking rate to a TTS speed factor."""
    if not rate or not mean_rate or mean_rate <= 0:
        return 1.0
    return round(_clamp(rate / mean_rate, SPEED_LOW, SPEED_HIGH), 3)


def compute_gain(energy: float, mean_energy: float) -> float:
    """Map a segment's relative loudness to a placement gain factor."""
    if energy <= 0 or mean_energy <= 0:
        return 1.0
    return round(_clamp(energy / mean_energy, GAIN_LOW, GAIN_HIGH), 3)
