"""Tests for new optimization metrics (ambience/reaction preservation)."""
import json
import numpy as np
import pytest
from pathlib import Path

from src.optimization.metrics import (
    ambience_preservation,
    reaction_preservation,
    compute_metrics,
)


@pytest.fixture
def workspace_with_ambience(tmp_path):
    """Create a workspace with original and final audio for ambience testing."""
    sr = 16000
    duration = 10.0
    n_samples = int(duration * sr)

    # Create media directory
    media_dir = tmp_path / "media"
    media_dir.mkdir()

    # Create output directory
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Original audio: speech + reactions + ambience
    t = np.linspace(0, duration, n_samples)
    orig = np.zeros(n_samples, dtype="float32")

    # Speech regions
    speech1 = (t >= 1.0) & (t < 3.0)
    orig[speech1] = 0.3 * np.sin(2 * np.pi * 500 * t[speech1])

    speech2 = (t >= 5.0) & (t < 7.0)
    orig[speech2] = 0.3 * np.sin(2 * np.pi * 600 * t[speech2])

    # Reaction region (laughter-like)
    reaction = (t >= 3.5) & (t < 4.5)
    orig[reaction] = 0.5 * np.random.randn(reaction.sum()) * 0.3

    # Ambience (low-level noise)
    orig += 0.02 * np.random.randn(n_samples)

    # Save original
    import soundfile as sf
    sf.write(str(media_dir / "original_audio.wav"), orig, sr)

    # Final audio: similar but with some differences
    final = orig.copy()
    # Add some noise to simulate imperfect reconstruction
    final += 0.01 * np.random.randn(n_samples)
    # Slightly reduce reaction intensity
    final[reaction] *= 0.8

    sf.write(str(output_dir / "final_audio.wav"), final, sr)

    # Create manifest
    manifest = [
        {"start": 1.0, "end": 3.0, "speaker": "speaker_01", "is_non_speech": False},
        {"start": 5.0, "end": 7.0, "speaker": "speaker_01", "is_non_speech": False},
    ]
    (tmp_path / "aligned_manifest.json").write_text(json.dumps(manifest))

    return tmp_path


def test_ambience_preservation_basic(workspace_with_ambience):
    """Test ambience preservation metric."""
    score, details = ambience_preservation(workspace_with_ambience)

    assert score is not None
    assert 0.0 <= score <= 1.0
    assert "energy_correlation" in details
    assert "non_speech_seconds" in details


def test_ambience_preservation_missing_files(tmp_path):
    """Test ambience preservation with missing files."""
    score, details = ambience_preservation(tmp_path)
    assert score is None


def test_reaction_preservation_basic(workspace_with_ambience):
    """Test reaction preservation metric."""
    score, details = reaction_preservation(workspace_with_ambience)

    assert score is not None
    assert 0.0 <= score <= 1.0
    assert "n_original" in details
    assert "n_final" in details
    assert "n_matched" in details


def test_reaction_preservation_missing_files(tmp_path):
    """Test reaction preservation with missing files."""
    score, details = reaction_preservation(tmp_path)
    assert score is None


def test_compute_metrics_includes_new_metrics(workspace_with_ambience):
    """Test that compute_metrics includes the new metrics."""
    result = compute_metrics(workspace_with_ambience)

    assert "ambience_preservation" in result.metrics
    assert "reaction_preservation" in result.metrics

    # Check that details are included
    assert "ambience_preservation" in result.details
    assert "reaction_preservation" in result.details


def test_metrics_weights_sum_to_one():
    """Test that default weights still sum to 1.0."""
    from src.optimization.metrics import DEFAULT_WEIGHTS
    total = sum(DEFAULT_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-6


def test_metrics_include_new_categories():
    """Test that new metric categories are in default weights."""
    from src.optimization.metrics import DEFAULT_WEIGHTS
    assert "ambience_preservation" in DEFAULT_WEIGHTS
    assert "reaction_preservation" in DEFAULT_WEIGHTS
