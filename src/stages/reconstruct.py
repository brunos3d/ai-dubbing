"""Stage 9 - Timeline reconstruction.

Place each translated segment at the original start time on a fresh silent
timeline, then mix with the original background to preserve ambience and
emotional non-verbal sounds.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf

from ..utils.audio import crossfade, read_wav, rms_normalize, write_wav
from ..utils.logging import get_logger, stage_banner
from ..utils.vram import free_vram, log_vram

LOG = get_logger("ai-dubbing.reconstruct")


def _load_mono(path: Path) -> tuple:
    """Read a wav as mono at its native sample rate.

    Returns ``(waveform, sample_rate)`` with ``waveform`` in PyTorch's
    ``(channels, samples)`` layout — mono files are ``(1, samples)``.
    """
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    if data.ndim == 2:
        if data.shape[1] > 1:
            data = data.mean(axis=1, keepdims=True)
        else:
            data = data.T  # (samples, 1) -> (1, samples)
    else:
        data = data[np.newaxis, :]
    return data.astype("float32", copy=False), int(sr)


def _overlay(silent: np.ndarray, seg: np.ndarray, start: int, end: int) -> np.ndarray:
    n = min(seg.shape[-1], silent.shape[-1] - start) if start < silent.shape[-1] else 0
    if n <= 0:
        return silent
    if end > silent.shape[-1]:
        end = silent.shape[-1]
    win = np.linspace(0.0, 1.0, max(1, min(64, n // 2)), dtype=seg.dtype)
    fade_in = np.ones_like(seg[..., :n])
    fade_in[..., : win.shape[0]] = win
    fade_out = np.ones_like(seg[..., :n])
    fade_out[..., -win.shape[0]:] = win[::-1]
    seg_eq = seg[..., :n] * fade_in * fade_out
    silent[..., start:start + n] = np.maximum(
        np.abs(silent[..., start:start + n]),
        seg_eq,
    )
    return silent


def _mix_additive(silent: np.ndarray, seg: np.ndarray, start: int) -> np.ndarray:
    n = seg.shape[-1]
    end = min(silent.shape[-1], start + n)
    if start >= silent.shape[-1] or end <= start:
        return silent
    seg_clip = seg[..., : end - start]
    silent[..., start:end] += seg_clip
    return silent


def reconstruct_timeline(
    aligned_manifest: List[Dict[str, Any]],
    background_path: Path,
    output_path: Path,
    target_sr: int = 24000,
    min_silence_floor: float = 0.02,
) -> Dict[str, Any]:
    bg_data, bg_sr = _load_mono(background_path)
    bg_n_samples = bg_data.shape[0] if bg_data.ndim >= 1 else 0
    bg_rms = float(np.sqrt(np.mean(bg_data**2))) if bg_n_samples > 0 else 0.0
    if bg_n_samples < 1000 or bg_rms < 1e-6:
        LOG.warning(
            f"Background {background_path} looks degenerate "
            f"(shape={bg_data.shape}, sr={bg_sr}, rms={bg_rms:.4f}); "
            f"reconstructing as speech-only timeline"
        )
        bg_data = np.zeros((1, target_sr), dtype="float32")
        bg_sr = target_sr
    else:
        if bg_sr != target_sr:
            import librosa  # type: ignore

            bg_data = librosa.resample(bg_data[0], orig_sr=bg_sr, target_sr=target_sr)[np.newaxis, :]
            bg_sr = target_sr

    end_samples = int(round(max(
        (a.get("end", a.get("start", 0.0)) for a in aligned_manifest),
        default=bg_data.shape[-1] / bg_sr,
    ) * bg_sr))
    end_samples = max(end_samples, bg_data.shape[-1])
    timeline_len = end_samples + int(0.5 * bg_sr)
    speech_timeline = np.zeros((1, timeline_len), dtype="float32")

    used_segments = 0
    for entry in aligned_manifest:
        if entry.get("alignment_method") == "passthrough" and not Path(entry["aligned_path"]).exists():
            continue
        path = Path(entry["aligned_path"])
        if not path.exists():
            path = Path(entry["path"])
        if not path.exists():
            continue
        seg_data, seg_sr = _load_mono(path)
        if seg_sr != bg_sr:
            import librosa  # type: ignore

            seg_data = librosa.resample(seg_data[0], orig_sr=seg_sr, target_sr=bg_sr)[np.newaxis, :]
            seg_sr = bg_sr
        start = int(round(float(entry.get("start", 0.0)) * bg_sr))
        end = int(round(float(entry.get("end", entry.get("start", 0.0) + seg_data.shape[-1] / bg_sr)) * bg_sr))
        target_dur = max(0.1, (end - start) / bg_sr)
        if seg_data.shape[-1] / bg_sr > target_dur * 1.4 and target_dur > 0.4:
            from ..utils.audio import time_stretch

            rate = (seg_data.shape[-1] / bg_sr) / target_dur
            seg_data = time_stretch(seg_data, rate=rate)
        speech_timeline = _mix_additive(speech_timeline, seg_data, start)
        used_segments += 1

    speech_timeline = np.clip(speech_timeline, -1.0, 1.0)
    write_wav(output_path, speech_timeline, bg_sr)
    return {
        "reconstructed_path": str(output_path),
        "duration_s": speech_timeline.shape[-1] / bg_sr,
        "num_segments_used": used_segments,
        "sample_rate": bg_sr,
        "background_path": str(background_path),
    }


class ReconstructStage:
    name = "reconstruct"
    inputs: List[str] = [
        "aligned_manifest.json",
        "media/background.wav",
    ]
    outputs: List[str] = ["output/reconstructed_speech.wav"]
    editable_outputs: List[str] = []
    derived_outputs: List[str] = ["output/reconstructed_speech.wav"]
    config_fields: List[str] = ["target_sr"]

    def __init__(self, workdir: Path, output_dir: Path, target_sr: int = 24000, subdir: str | None = None):
        self.workdir = Path(workdir)
        if subdir:
            self.workdir = self.workdir / subdir
        self.output_dir = Path(output_dir)
        self.target_sr = target_sr
        self.subdir = subdir

    def output_paths(self) -> List[Path]:
        return [self.output_dir / "reconstructed_speech.wav"]

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        stage_banner(LOG, 8, 11, "Reconstruction")
        aligned_manifest_path = Path(context["manifest_path"])
        background_path = Path(context["background_path"])
        LOG.info(f"Reading aligned manifest from {aligned_manifest_path}")
        aligned_manifest = json.loads(aligned_manifest_path.read_text(encoding="utf-8"))
        if aligned_manifest and "aligned_path" not in aligned_manifest[0]:
            LOG.error(f"Manifest entries lack aligned_path. Sample: {aligned_manifest[0]}")
        output_path = self.output_dir / "reconstructed_speech.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        info = reconstruct_timeline(
            aligned_manifest,
            background_path,
            output_path,
            target_sr=self.target_sr,
        )
        LOG.info(f"Reconstructed -> {output_path} ({info['duration_s']:.2f}s)")
        free_vram()
        return info
