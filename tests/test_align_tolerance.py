"""Tests for align's absolute-delta floor (Failure #3, downstream cost).

Word-level speaker splitting produces more, shorter segments. A purely
relative tolerance (10%) time-stretches a 0.7s clip whenever it is >0.07s
off — an inaudible error that does not justify a lossy resampling pass. An
absolute floor skips those pointless micro-corrections so fewer downstream
stretches are needed while real (audible) drift is still fixed.
"""
from __future__ import annotations

from src.stages.align import _needs_correction


def test_small_absolute_error_is_skipped():
    # 0.7s clip, 0.08s off -> 11% (over relative tol) but < absolute floor.
    assert _needs_correction(0.78, 0.70, tol=0.10, min_abs_s=0.12) is False


def test_large_absolute_error_is_corrected():
    # 6.0s clip, 1.5s off -> well over both thresholds.
    assert _needs_correction(7.5, 6.0, tol=0.10, min_abs_s=0.12) is True


def test_small_relative_error_is_skipped_even_if_above_floor():
    # 30s clip, 0.2s off -> over the absolute floor but only 0.7% relative.
    assert _needs_correction(30.2, 30.0, tol=0.10, min_abs_s=0.12) is False


def test_both_thresholds_must_be_exceeded():
    # Exactly at the boundary: 1.0s clip, 0.11s off -> 11% relative but
    # below the 0.12s floor -> skip.
    assert _needs_correction(1.11, 1.0, tol=0.10, min_abs_s=0.12) is False
    # Bump the error past the floor -> correct.
    assert _needs_correction(1.20, 1.0, tol=0.10, min_abs_s=0.12) is True
