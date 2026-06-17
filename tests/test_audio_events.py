"""Tests for audio event classification."""
import numpy as np
import pytest
from pathlib import Path

from src.utils.audio_events import (
    AudioEvent,
    classify_audio_layers,
    detect_reactions,
    detect_ambience_regions,
    save_event_manifest,
    load_event_manifest,
)


@pytest.fixture
def sample_audio():
    """Create a sample audio signal with speech and non-speech regions."""
    sr = 16000
    duration = 10.0
    n_samples = int(duration * sr)
    audio = np.zeros(n_samples, dtype="float32")

    # Add some "speech" regions (mid-frequency tones)
    t = np.linspace(0, duration, n_samples)
    # Speech at 1-3 seconds
    speech_mask = (t >= 1.0) & (t < 3.0)
    audio[speech_mask] = 0.3 * np.sin(2 * np.pi * 500 * t[speech_mask])

    # Speech at 5-7 seconds
    speech_mask = (t >= 5.0) & (t < 7.0)
    audio[speech_mask] = 0.3 * np.sin(2 * np.pi * 600 * t[speech_mask])

    # Add some "reaction" regions (higher energy, broadband)
    # Reaction at 3.5-4.5 seconds (laughter-like)
    reaction_mask = (t >= 3.5) & (t < 4.5)
    audio[reaction_mask] = 0.5 * np.random.randn(reaction_mask.sum()) * 0.3

    # Add some ambience (low-level noise)
    audio += 0.02 * np.random.randn(n_samples)

    return audio, sr


@pytest.fixture
def speech_segments():
    """Speech segments matching the sample audio."""
    return [
        {"start": 1.0, "end": 3.0, "speaker": "speaker_01"},
        {"start": 5.0, "end": 7.0, "speaker": "speaker_01"},
    ]


def test_audio_event_to_dict():
    """Test AudioEvent serialization."""
    event = AudioEvent(
        event_type="laughter",
        start=1.5,
        end=2.5,
        confidence=0.8,
        intensity=0.7,
    )
    d = event.to_dict()
    assert d["event_type"] == "laughter"
    assert d["start"] == 1.5
    assert d["end"] == 2.5
    assert d["confidence"] == 0.8
    assert d["intensity"] == 0.7


def test_detect_reactions(sample_audio, speech_segments):
    """Test reaction detection."""
    audio, sr = sample_audio
    reactions = detect_reactions(audio, sr, speech_segments, min_duration=0.2)

    # Should detect the reaction region around 3.5-4.5 seconds
    assert len(reactions) > 0

    # Check that detected reactions are in non-speech regions
    for r in reactions:
        # Should not overlap with speech segments
        for seg in speech_segments:
            overlap_start = max(r.start, seg["start"])
            overlap_end = min(r.end, seg["end"])
            assert overlap_end <= overlap_start + 0.1  # Allow small overlap


def test_detect_ambience_regions(sample_audio, speech_segments):
    """Test ambience region detection."""
    audio, sr = sample_audio
    reactions = detect_reactions(audio, sr, speech_segments, min_duration=0.2)
    ambience = detect_ambience_regions(audio, sr, speech_segments, reactions, min_duration=0.3)

    # Should detect ambience regions in non-speech, non-reaction areas
    assert len(ambience) > 0

    # Check that ambience regions don't overlap with speech
    for a in ambience:
        for seg in speech_segments:
            overlap_start = max(a.start, seg["start"])
            overlap_end = min(a.end, seg["end"])
            assert overlap_end <= overlap_start + 0.1


def test_classify_audio_layers(sample_audio, speech_segments):
    """Test full audio layer classification."""
    audio, sr = sample_audio
    layers = classify_audio_layers(audio, sr, speech_segments)

    assert "speech" in layers
    assert "reactions" in layers
    assert "ambience" in layers

    # Speech should match input
    assert len(layers["speech"]) == len(speech_segments)


def test_save_load_manifest(tmp_path, sample_audio, speech_segments):
    """Test manifest save/load roundtrip."""
    audio, sr = sample_audio
    events = classify_audio_layers(audio, sr, speech_segments)

    manifest_path = tmp_path / "events.json"
    save_event_manifest(events, manifest_path)

    loaded = load_event_manifest(manifest_path)

    assert len(loaded["speech"]) == len(events["speech"])
    assert len(loaded["reactions"]) == len(events["reactions"])
    assert len(loaded["ambience"]) == len(events["ambience"])


def test_empty_audio():
    """Test classification with empty/minimal audio."""
    audio = np.zeros(100, dtype="float32")
    sr = 16000
    speech_segments = []

    layers = classify_audio_layers(audio, sr, speech_segments)
    assert len(layers["speech"]) == 0
    assert len(layers["reactions"]) == 0


def test_all_speech():
    """Test classification when entire audio is speech."""
    sr = 16000
    duration = 5.0
    n_samples = int(duration * sr)
    audio = 0.3 * np.random.randn(n_samples).astype("float32")

    speech_segments = [{"start": 0.0, "end": duration}]

    layers = classify_audio_layers(audio, sr, speech_segments)
    assert len(layers["speech"]) == 1
    # Should have no reactions or ambience since entire audio is speech
    assert len(layers["reactions"]) == 0
