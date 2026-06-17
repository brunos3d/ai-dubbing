"""OmniVoice backend — the default synthesizer.

This wraps the exact OmniVoice load + duration-fitting logic that previously
lived in ``stages/generate.py``, so behaviour is unchanged: VRAM-aware device
mapping, iterative speed fitting, and prosody-derived base speed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..utils.audio import read_wav
from ..utils.logging import get_logger
from ..utils.vram import free_vram
from .base import (
    AudioSegment,
    SpeakerProfile,
    VoiceSynthesizer,
    fit_by_speed,
    language_name,
)

LOG = get_logger("ai-dubbing.tts.omnivoice")

_DEFAULT_SR = 24000


def _coerce_ref_audio(path: str) -> tuple:
    """Read reference audio into a ``(waveform_tensor, sample_rate)`` tuple."""
    data, sr = read_wav(path, dtype="float32", always_2d=True)
    if data.shape[0] > 1:
        data = data.mean(axis=0, keepdims=True)
    wav = torch.from_numpy(data.astype("float32"))
    return wav, int(sr)


class OmniVoiceSynthesizer(VoiceSynthesizer):
    """k2-fsa/OmniVoice voice-cloning backend (pipeline default)."""

    backend_id = "omnivoice"

    def __init__(
        self,
        *,
        language: Optional[str] = None,
        device: str = "cuda",
        model_id: str = "k2-fsa/OmniVoice",
        offload_dir: str = "/tmp/opencode/omnivoice_offload",
        tolerance: float = 0.10,
        max_speed: float = 1.35,
        max_fit_iters: int = 2,
        model_obj: Any = None,
    ) -> None:
        super().__init__(
            language=language, device=device, tolerance=tolerance,
            max_speed=max_speed, max_fit_iters=max_fit_iters,
        )
        self.model_id = model_id
        self.offload_dir = offload_dir
        # ``model_obj`` lets tests inject a fake OmniVoice without GPU/network.
        self._model = model_obj
        self._model_sr = int(getattr(model_obj, "sampling_rate", _DEFAULT_SR)) if model_obj is not None else _DEFAULT_SR
        if model_obj is not None:
            self._loaded = True

    # -- lifecycle ---------------------------------------------------------

    def _load(self) -> None:
        from omnivoice import OmniVoice

        import os
        Path(self.offload_dir).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        try:
            free, _ = torch.cuda.mem_get_info()
            free_gb = free / (1024 ** 3)
        except Exception:
            free_gb = 0.0

        LOG.info(f"Loading OmniVoice ({self.model_id}) on {self.device}")
        if free_gb < 3.5 and self.device.startswith("cuda"):
            gpu_cap = min(0.9, max(0.6, free_gb - 1.7))
            LOG.info(
                f"Only {free_gb:.2f} GB free VRAM; device_map='auto' with offload "
                f"(GPU cap {gpu_cap:.1f} GiB)"
            )
            try:
                self._model = OmniVoice.from_pretrained(
                    self.model_id,
                    device_map="auto",
                    max_memory={0: f"{gpu_cap:.1f}GiB", "cpu": "30GiB"},
                    dtype=torch.float16,
                    offload_folder=self.offload_dir,
                )
            except Exception as exc:
                LOG.warning(f"auto device map with max_memory failed: {exc}")
        if self._model is None:
            self._model = OmniVoice.from_pretrained(
                self.model_id, device_map=self.device, dtype=torch.float16,
            )
        self._model_sr = int(getattr(self._model, "sampling_rate", _DEFAULT_SR))

    def _close(self) -> None:
        try:
            del self._model
        except Exception:
            pass
        self._model = None
        free_vram()

    @property
    def sample_rate(self) -> int:
        return self._model_sr

    # -- synthesis ---------------------------------------------------------

    def generate(
        self,
        text: str,
        speaker_profile: SpeakerProfile,
        target_duration: Optional[float] = None,
        *,
        base_speed: float = 1.0,
    ) -> AudioSegment:
        if self._model is None:
            self.load()
        wav, sr = _coerce_ref_audio(speaker_profile.reference_wav)
        lang = language_name(self.language)
        ref_text = speaker_profile.reference_text or None

        def _synth(speed: float) -> np.ndarray:
            try:
                audios = self._model.generate(
                    text=text, language=lang, ref_audio=(wav, sr),
                    ref_text=ref_text, speed=float(speed),
                )
            except Exception as exc:  # noqa: BLE001 - retry without conditioning
                LOG.warning(f"OmniVoice generate failed ({exc}); retrying minimally")
                audios = self._model.generate(
                    text=text, language=lang, ref_audio=(wav, sr), ref_text=ref_text,
                )
            if not audios:
                raise RuntimeError("OmniVoice returned no audio")
            a = audios[0]
            if isinstance(a, torch.Tensor):
                a = a.detach().cpu().numpy()
            return np.asarray(a, dtype=np.float32).reshape(-1)

        seg = fit_by_speed(
            _synth, self.sample_rate, target_duration, base_speed=base_speed,
            tol=self.tolerance, max_speed=self.max_speed, max_iters=self.max_fit_iters,
        )
        seg.backend = self.backend_id
        return seg
