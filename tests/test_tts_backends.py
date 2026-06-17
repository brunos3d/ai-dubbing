"""Unit tests for the pluggable TTS layer (VoiceSynthesizer)."""
from __future__ import annotations

import numpy as np
import pytest

from src.tts import (
    AudioSegment,
    SpeakerProfile,
    VoiceSynthesizer,
    available_backends,
    is_known_backend,
    make_synthesizer,
)
from src.tts.base import fit_by_speed, next_speed
from src.tts.f5tts import F5TTSSynthesizer
from src.tts.omnivoice import OmniVoiceSynthesizer


# --- factory / selection --------------------------------------------------

def test_default_backend_is_omnivoice():
    s = make_synthesizer()
    assert isinstance(s, OmniVoiceSynthesizer)
    assert s.backend_id == "omnivoice"


def test_factory_selects_each_backend():
    assert isinstance(make_synthesizer("omnivoice"), OmniVoiceSynthesizer)
    assert isinstance(make_synthesizer("f5tts"), F5TTSSynthesizer)
    # Case/spacing tolerant.
    assert isinstance(make_synthesizer("  F5TTS "), F5TTSSynthesizer)


def test_available_and_known_backends():
    assert available_backends()[0] == "omnivoice"  # default first
    assert set(available_backends()) == {"omnivoice", "f5tts"}
    assert is_known_backend("f5tts")
    assert not is_known_backend("nope")


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown TTS backend"):
        make_synthesizer("does-not-exist")


def test_factory_passes_kwargs_through():
    s = make_synthesizer("omnivoice", language="pt", max_speed=1.5, device="cpu")
    assert s.language == "pt" and s.max_speed == 1.5 and s.device == "cpu"


# --- shared fitting logic -------------------------------------------------

def test_next_speed_overrun_and_underrun():
    assert next_speed(6.0, 4.5, 1.0, 1.35, 0.10) == pytest.approx(1.333, abs=0.02)
    assert next_speed(3.0, 4.5, 1.0, 1.35, 0.10) is None  # underrun accepted
    assert next_speed(12.0, 4.0, 1.0, 1.35, 0.10) == 1.35  # capped


def test_fit_by_speed_converges():
    # Synthetic backend whose length scales as base_len / speed.
    def synth(speed):
        return np.zeros(int(6.0 / speed * 24000), dtype="float32")

    seg = fit_by_speed(synth, 24000, slot_s=4.5, base_speed=1.0, max_iters=2)
    assert isinstance(seg, AudioSegment)
    assert seg.speed > 1.0 and seg.duration_s < 6.0
    assert seg.iterations >= 2


# --- backend generation via injected fake models --------------------------

class _FakeOmni:
    sampling_rate = 24000

    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        speed = kwargs.get("speed") or 1.0
        return [np.zeros(int(5.0 / speed * 24000), dtype="float32")]


class _FakeF5:
    target_sample_rate = 24000

    def __init__(self):
        self.calls = []

    def infer(self, ref_file, ref_text, gen_text, speed=1.0, **kwargs):
        self.calls.append((gen_text, speed))
        return (np.zeros(int(5.0 / speed * 24000), dtype="float32"), 24000, None)


def _profile(tmp_path):
    import soundfile as sf
    ref = tmp_path / "ref.wav"
    sf.write(str(ref), np.zeros(16000, dtype="float32"), 16000)
    return SpeakerProfile("speaker_01", str(ref), "reference transcript")


def test_omnivoice_generate_returns_tagged_segment(tmp_path):
    fake = _FakeOmni()
    s = OmniVoiceSynthesizer(language="pt", model_obj=fake)
    seg = s.generate("Olá mundo", _profile(tmp_path), target_duration=3.0)
    assert seg.backend == "omnivoice"
    assert seg.sample_rate == 24000
    assert seg.num_samples > 0
    assert fake.calls  # model was actually called
    # The OmniVoice call carries text/language/ref/speed.
    assert fake.calls[0]["text"] == "Olá mundo"
    assert fake.calls[0]["language"] == "Portuguese"


def test_f5tts_generate_returns_tagged_segment(tmp_path):
    fake = _FakeF5()
    s = F5TTSSynthesizer(language="pt", model_obj=fake)
    seg = s.generate("Olá mundo", _profile(tmp_path), target_duration=3.0)
    assert seg.backend == "f5tts"
    assert seg.sample_rate == 24000
    assert seg.num_samples > 0
    assert fake.calls and fake.calls[0][0] == "Olá mundo"


def test_f5tts_missing_package_raises_clear_error():
    """Selecting F5 without the package gives an actionable message on load."""
    s = F5TTSSynthesizer(language="pt")  # no injected model
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("f5_tts"):
            raise ModuleNotFoundError("No module named 'f5_tts'")
        return real_import(name, *a, **k)

    builtins.__import__ = fake_import
    try:
        with pytest.raises(RuntimeError, match="pip install f5-tts"):
            s.load()
    finally:
        builtins.__import__ = real_import


def test_pipeline_depends_only_on_abstraction():
    """Both concrete backends honour the VoiceSynthesizer contract."""
    for backend in available_backends():
        s = make_synthesizer(backend)
        assert isinstance(s, VoiceSynthesizer)
        assert hasattr(s, "generate") and hasattr(s, "load") and hasattr(s, "close")
