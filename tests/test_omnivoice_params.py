"""Tests for the expanded OmniVoice parameters and ending quality improvements."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.stages.generate import GenerateStage
from src.tts import SpeakerProfile
from src.tts.omnivoice import OmniVoiceSynthesizer
from src.utils.config import save_pipeline_defaults


class _FakeOmni:
    sampling_rate = 24000

    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        speed = kwargs.get("speed") or 1.0
        # Return ones so we can test the fade-out effect.
        return [torch.ones(int(5.0 / speed * 24000), dtype=torch.float32)]


def _profile(tmp_path):
    import soundfile as sf
    ref = tmp_path / "ref.wav"
    sf.write(str(ref), np.zeros(16000, dtype="float32"), 16000)
    return SpeakerProfile("speaker_01", str(ref), "reference transcript")


def test_omnivoice_param_propagation(tmp_path):
    fake = _FakeOmni()
    s = OmniVoiceSynthesizer(
        num_step=40,
        guidance_scale=3.0,
        t_shift=0.2,
        postprocess_output=False,
        audio_chunk_duration=20.0,
        model_obj=fake,
    )
    s.generate("Test", _profile(tmp_path))
    
    assert fake.calls
    c = fake.calls[0]
    assert c["num_step"] == 40
    assert c["guidance_scale"] == 3.0
    assert c["t_shift"] == 0.2
    assert c["postprocess_output"] is False
    assert c["audio_chunk_duration"] == 20.0


def test_omnivoice_tail_padding_and_fade(tmp_path):
    fake = _FakeOmni()
    # 0.1s padding at 24000Hz is 2400 samples.
    # FakeOmni returns 5.0s which is 120000 samples.
    s = OmniVoiceSynthesizer(tail_padding_s=0.1, model_obj=fake)
    seg = s.generate("Hello", _profile(tmp_path))
    
    # 120000 + 2400 = 122400
    assert seg.num_samples == 122400
    
    # The padding zone (last 100ms) should be 0.0.
    assert seg.samples[-1] == 0.0
    assert seg.samples[121000] == 0.0
    
    # The speech zone before fade should be 1.0.
    # Fade starts at 120000 - 1200 (50ms) = 118800.
    assert seg.samples[118799] == 1.0
    
    # In the middle of the fade (e.g. 25ms into the 50ms fade).
    # fade_curve = linspace(1.0, 0.0, 1200).
    # At index 118800 + 600 = 119400.
    assert 0.4 < seg.samples[119400] < 0.6


def test_generate_stage_wires_omnivoice_params(tmp_path):
    stage = GenerateStage(
        tmp_path,
        num_step=48,
        guidance_scale=4.0,
        tail_padding_s=0.2,
    )
    # Mock the internal make_synthesizer to return a synthesizer with an injected fake model.
    fake = _FakeOmni()
    synth = OmniVoiceSynthesizer(
        num_step=stage.num_step,
        guidance_scale=stage.guidance_scale,
        tail_padding_s=stage.tail_padding_s,
        model_obj=fake,
    )
    
    # We test the stage's config_fields first.
    assert "num_step" in stage.config_fields
    assert "guidance_scale" in stage.config_fields
    assert "tail_padding_s" in stage.config_fields
    
    # Test property assignment in __init__
    assert stage.num_step == 48
    assert stage.guidance_scale == 4.0
    assert stage.tail_padding_s == 0.2


def test_config_registry_includes_omnivoice_block(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setattr("src.utils.config.config_root", lambda: config_dir)
    
    defaults = {
        "samples.target_seconds": 12.0,
        "omnivoice": {
            "num_step": 32,
            "postprocess_output": True,
            "tail_padding_s": 0.15
        }
    }
    save_pipeline_defaults(defaults)
    
    from src.utils.config import load_pipeline_defaults
    loaded = load_pipeline_defaults()
    assert loaded["omnivoice"]["num_step"] == 32
    assert loaded["omnivoice"]["tail_padding_s"] == 0.15


def test_to_stage_overrides_maps_omnivoice_to_generate():
    from src.optimization.parameter_space import to_stage_overrides
    config = {
        "samples.target_seconds": 13.0,
        "omnivoice": {
            "num_step": 64,
            "tail_padding_s": 0.2
        },
        "omnivoice.guidance_scale": 4.5
    }
    overrides = to_stage_overrides(config)
    
    assert overrides["samples"]["target_seconds"] == 13.0
    # Both nested and flat omnivoice keys should map to 'generate'.
    assert overrides["generate"]["num_step"] == 64
    assert overrides["generate"]["tail_padding_s"] == 0.2
    assert overrides["generate"]["guidance_scale"] == 4.5
