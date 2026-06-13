"""Audio I/O helpers wrapping soundfile + numpy."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf


def read_wav(path: str | Path, dtype: str = "float32", always_2d: bool = True) -> Tuple[np.ndarray, int]:
    """Read a wav file. Returns (waveform, sample_rate).

    waveform shape: (channels, samples) when always_2d True, else (samples,)
    """
    data, sr = sf.read(str(path), dtype=dtype, always_2d=always_2d)
    if always_2d and data.ndim == 1:
        data = data[np.newaxis, :]
    elif always_2d and data.ndim == 2:
        data = data.T
    return data, sr


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int, subtype: str = "PCM_16") -> None:
    """Write a wav file. Audio is (channels, samples) float32 in [-1, 1]."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(audio)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim == 2:
        arr = arr.T
    arr = np.clip(arr, -1.0, 1.0)
    sf.write(str(path), arr, sample_rate, subtype=subtype)


def to_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data
    return data.mean(axis=0)


def peak_normalize(data: np.ndarray, target_db: float = -1.0, eps: float = 1e-9) -> np.ndarray:
    """Peak-normalize audio so the loudest sample sits at `target_db`."""
    if data.size == 0:
        return data
    peak = float(np.max(np.abs(data)))
    if peak < eps:
        return data
    target_amp = 10.0 ** (target_db / 20.0)
    return data * (target_amp / peak)


def rms_normalize(data: np.ndarray, target_dbfs: float = -20.0, eps: float = 1e-9) -> np.ndarray:
    """RMS-normalize audio to a target loudness in dBFS."""
    if data.size == 0:
        return data
    rms = float(np.sqrt(np.mean(np.square(data))))
    if rms < eps:
        return data
    target_rms = 10.0 ** (target_dbfs / 20.0)
    return data * (target_rms / rms)


def time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """Phase-vocoder-ish time stretch using librosa if available, else fallback.

    `rate > 1.0` shortens the audio; `rate < 1.0` lengthens it.
    """
    if rate <= 0 or abs(rate - 1.0) < 1e-3:
        return audio
    try:
        import librosa  # type: ignore

        if audio.ndim == 1:
            return librosa.effects.time_stretch(audio, rate=rate)
        out = np.stack(
            [librosa.effects.time_stretch(ch, rate=rate) for ch in audio], axis=0
        )
        return out
    except ImportError:
        return _simple_time_stretch(audio, rate)


def _simple_time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """Linear-interpolation stretch used as a fallback when librosa is missing."""
    if audio.ndim == 1:
        return _stretch_1d(audio, rate)
    return np.stack([_stretch_1d(ch, rate) for ch in audio], axis=0)


def _stretch_1d(audio: np.ndarray, rate: float) -> np.ndarray:
    n_in = audio.shape[0]
    n_out = max(1, int(round(n_in / rate)))
    xp = np.linspace(0, n_in - 1, n_out)
    return np.interp(xp, np.arange(n_in), audio).astype(audio.dtype, copy=False)


def crossfade(a: np.ndarray, b: np.ndarray, fade_seconds: float, sample_rate: int) -> np.ndarray:
    """Crossfade two mono or multichannel audio chunks of equal channel count."""
    if a.size == 0:
        return b
    if b.size == 0:
        return a
    fade_n = max(1, int(round(fade_seconds * sample_rate)))
    fade_n = min(fade_n, a.shape[-1], b.shape[-1])
    if fade_n <= 0:
        return np.concatenate([a, b], axis=-1)
    fade_in = np.linspace(0.0, 1.0, fade_n, dtype=a.dtype)
    fade_out = 1.0 - fade_in
    head = a[..., :-fade_n] if a.shape[-1] > fade_n else np.zeros_like(a[..., :0])
    tail_a = a[..., -fade_n:] * fade_out
    tail_b = b[..., :fade_n] * fade_in
    cross = tail_a + tail_b
    rest = b[..., fade_n:] if b.shape[-1] > fade_n else np.zeros_like(b[..., :0])
    return np.concatenate([head, cross, rest], axis=-1)


def compute_snr_db(noise: np.ndarray, eps: float = 1e-9) -> float:
    """Estimate SNR in dB using the loudest 20% vs quietest 20% of the signal."""
    if noise.size == 0:
        return 0.0
    abs_signal = np.abs(noise.flatten())
    if abs_signal.size < 4:
        return 0.0
    abs_sorted = np.sort(abs_signal)
    q = max(1, abs_sorted.size // 5)
    noise_floor = float(np.mean(abs_sorted[:q]))
    peak = float(np.mean(abs_sorted[-q:]))
    if noise_floor < eps:
        return 60.0
    return 20.0 * math.log10(peak / noise_floor)
