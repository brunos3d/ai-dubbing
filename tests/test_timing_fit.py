"""Tests for iterative duration-fitting at synthesis time (Objective #3).

Timing becomes a generation *constraint*: synthesize, measure the actual
output, and if it overruns its slot, raise OmniVoice's speed and regenerate
(bounded, to keep naturalness) instead of relying on a post-hoc resample.
Underruns are accepted as natural pauses (consistent with asymmetric align).
"""
from __future__ import annotations

import numpy as np

from src.stages.generate import _next_speed, generate_fitted


def test_no_change_when_inside_slot():
    # Fits comfortably -> no second pass.
    assert _next_speed(measured=4.0, slot=4.5, cur_speed=1.0, max_speed=1.35, tol=0.10) is None


def test_no_change_for_underrun():
    # Shorter than the slot -> leave it (natural pause), don't slow down.
    assert _next_speed(measured=3.0, slot=4.5, cur_speed=1.0, max_speed=1.35, tol=0.10) is None


def test_speed_up_on_overrun():
    # 6s in a 4.5s slot -> need ~1.33x.
    ns = _next_speed(measured=6.0, slot=4.5, cur_speed=1.0, max_speed=1.35, tol=0.10)
    assert ns is not None
    assert 1.3 <= ns <= 1.35


def test_speed_capped_for_naturalness():
    # Huge overrun -> clamp at max_speed, never chipmunk.
    ns = _next_speed(measured=12.0, slot=4.0, cur_speed=1.0, max_speed=1.35, tol=0.10)
    assert ns == 1.35


def test_no_change_when_already_at_cap():
    # Already at the cap and still over -> nothing more to do.
    assert _next_speed(measured=6.0, slot=4.0, cur_speed=1.35, max_speed=1.35, tol=0.10) is None


class _FakeModel:
    """Synthesises white noise whose length scales as base_len / speed."""

    def __init__(self, base_len_s, sr=24000):
        self.base_len_s = base_len_s
        self.sampling_rate = sr
        self.calls = []

    def generate(self, **kwargs):
        speed = kwargs.get("speed") or 1.0
        n = int(self.base_len_s / speed * self.sampling_rate)
        self.calls.append(speed)
        return [np.zeros(n, dtype="float32")]


def test_generate_fitted_converges_into_slot():
    model = _FakeModel(base_len_s=6.0)  # natural length 6s
    audio, sr, speed, dur, iters = generate_fitted(
        model, text="oi", language="Portuguese", ref_audio=(np.zeros(10), 24000),
        ref_text=None, slot_s=4.5, base_speed=1.0, tol=0.10, max_speed=1.35, max_iters=2,
    )
    # Started at 6s, sped up toward the 4.5s slot; ends within cap.
    assert dur < 6.0
    assert speed > 1.0
    assert len(model.calls) >= 2  # at least one corrective pass


def test_generate_fitted_single_pass_when_fits():
    model = _FakeModel(base_len_s=4.0)
    audio, sr, speed, dur, iters = generate_fitted(
        model, text="oi", language="Portuguese", ref_audio=(np.zeros(10), 24000),
        ref_text=None, slot_s=4.5, base_speed=1.0, tol=0.10, max_speed=1.35, max_iters=2,
    )
    assert len(model.calls) == 1  # fit on the first try
    assert speed == 1.0
