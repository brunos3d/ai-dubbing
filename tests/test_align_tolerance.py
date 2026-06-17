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


def test_moderate_underrun_is_tolerated_as_natural_gap():
    """A dub shorter than its slot leaves a natural pause; don't slow it.

    Real case: 6.28s slot, 5.20s dub (1.08s / 17% underrun) — over the
    overrun thresholds, but under the permissive underrun thresholds, so it
    is left alone instead of being dragged out.
    """
    assert _needs_correction(5.20, 6.28, tol=0.10, min_abs_s=0.12) is False


def test_large_underrun_is_still_corrected():
    # 4.0s slot, 1.0s dub -> 75% underrun: too big a hole, expand it.
    assert _needs_correction(1.0, 4.0, tol=0.10, min_abs_s=0.12) is True


def test_overrun_still_corrected_at_tight_threshold():
    # 4.0s slot, 4.6s dub -> 15% overrun, 0.6s absolute: bleeds into next
    # turn, must be compressed.
    assert _needs_correction(4.6, 4.0, tol=0.10, min_abs_s=0.12) is True
