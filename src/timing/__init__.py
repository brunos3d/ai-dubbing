"""Timing-aware dubbing primitives (spec 2026-06-16, Phase 2 & 5).

This package holds the CPU-only, dependency-light building blocks that let
the pipeline reason about *how long a translation will take to speak* before
synthesizing it, and how the source line was *delivered*:

* :mod:`src.timing.duration` — language-aware spoken-duration estimation.
* :mod:`src.timing.score` — translation-candidate timing score + selection.
* :mod:`src.timing.prosody` — per-segment prosody descriptors from audio.

None of these import torch or hit the network; they are pure functions that
are fully unit-testable and safe on a CPU-only machine.
"""
from __future__ import annotations

from .duration import DurationEstimate, estimate_duration, estimate_syllables

__all__ = ["DurationEstimate", "estimate_duration", "estimate_syllables"]
