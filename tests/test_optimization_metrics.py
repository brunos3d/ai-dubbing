"""Tests for the optimization metrics, on a synthetic mini-workspace."""
import json
from pathlib import Path

import numpy as np
import pytest

from src.optimization import metrics as M
from src.utils.audio import write_wav


def _tone(freq, dur, sr=24000, amp=0.3, decay=True):
    t = np.linspace(0, dur, int(dur * sr), endpoint=False)
    sig = amp * np.sin(2 * np.pi * freq * t).astype("float32")
    if decay:  # natural fade-out → good "ending"
        env = np.linspace(1.0, 0.0, sig.shape[0]).astype("float32")
        sig = sig * env
    return sig, sr


def _build_workspace(tmp_path: Path):
    sr = 24000
    seg_dir = tmp_path / "aligned_segments"
    seg_dir.mkdir(parents=True)
    prof = tmp_path / "speaker_profiles" / "speaker_01" / "primary"
    prof.mkdir(parents=True)
    (tmp_path / "media").mkdir()
    (tmp_path / "output").mkdir()

    # Speaker reference (220 Hz tone).
    ref, _ = _tone(220, 3.0, decay=False)
    write_wav(prof / "reference.wav", ref, sr)

    manifest = []
    timeline = np.zeros(int(10 * sr), dtype="float32")
    for i in range(4):
        start = i * 2.0
        a, _ = _tone(220, 1.5)  # same speaker timbre
        path = seg_dir / f"segment_{i+1:04d}.wav"
        write_wav(path, a, sr)
        s0 = int(start * sr)
        timeline[s0:s0 + a.shape[0]] += a
        manifest.append({
            "index": i, "speaker": "speaker_01", "is_non_speech": False,
            "start": start, "end": start + 1.5,
            "target_duration": 1.5, "original_duration": 1.5,
            "generated_duration": 1.5, "aligned_duration": 1.5,
            "aligned_path": str(path), "path": str(path),
            "alignment_method": "passthrough",
            "prosody": {"energy_db": M._db(float(np.sqrt(np.mean(a ** 2))))},
        })
    (tmp_path / "aligned_manifest.json").write_text(json.dumps(manifest))
    write_wav(tmp_path / "output" / "reconstructed_speech.wav", timeline, sr)
    write_wav(tmp_path / "media" / "speech.wav", timeline.copy(), sr)
    return tmp_path, manifest


def test_compute_metrics_on_synthetic_workspace(tmp_path):
    root, _ = _build_workspace(tmp_path)
    res = M.compute_metrics(root)
    assert res.composite is not None
    assert 0.0 <= res.composite <= 1.0
    for name, val in res.metrics.items():
        assert val is None or 0.0 <= val <= 1.0
    # Perfect timing & identical reconstruction should score very high.
    assert res.metrics["timing_accuracy"] > 0.95
    assert res.metrics["reconstruction_quality"] > 0.9
    assert res.metrics["speaker_similarity"] > 0.8


def test_timing_accuracy_penalizes_overrun():
    manifest = [
        {"is_non_speech": False, "target_duration": 1.0, "aligned_duration": 2.0},
    ]
    score, _ = M.timing_accuracy(manifest)
    assert score is not None and score < 0.1  # 100% error → ~0


def test_slot_fit_ignores_underrun():
    over = [{"is_non_speech": False, "target_duration": 1.0, "aligned_duration": 1.5}]
    under = [{"is_non_speech": False, "target_duration": 1.0, "aligned_duration": 0.5}]
    s_over, _ = M.slot_fit(over)
    s_under, _ = M.slot_fit(under)
    assert s_under == 1.0  # underrun not penalized
    assert s_over < 1.0


def test_continuity_detects_collision():
    manifest = [
        {"start": 0.0, "aligned_duration": 2.0},   # ends at 2.0
        {"start": 1.0, "aligned_duration": 1.0},   # starts at 1.0 → overlap
    ]
    score, det = M.continuity(manifest)
    assert det["n_collisions"] == 1
    assert score < 1.0


def test_score_from_metrics_renormalizes_over_available():
    metrics = {"timing_accuracy": 1.0, "slot_fit": None, "continuity": 0.0}
    # weights present for all three; None one is dropped and weights renormalize.
    score = M.score_from_metrics(metrics, {"timing_accuracy": 0.5, "slot_fit": 0.3, "continuity": 0.5})
    expected = (0.5 * 1.0 + 0.5 * 0.0) / (0.5 + 0.5)
    assert abs(score - expected) < 1e-9


def test_missing_artifacts_yield_none_not_crash(tmp_path):
    res = M.compute_metrics(tmp_path)  # empty dir
    assert res.composite is None
    assert all(v is None for v in res.metrics.values())
