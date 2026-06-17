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
        # Decoding
        num_step: int = 32,
        denoise: bool = True,
        guidance_scale: float = 2.0,
        t_shift: float = 0.1,
        # Sampling
        position_temperature: float = 5.0,
        class_temperature: float = 0.0,
        # Pre/Post Processing
        preprocess_prompt: bool = True,
        postprocess_output: bool = True,
        tail_padding_s: float = 0.1,
        audio_chunk_duration: float = 15.0,
        audio_chunk_threshold: float = 30.0,
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

        self.num_step = num_step
        self.denoise = denoise
        self.guidance_scale = guidance_scale
        self.t_shift = t_shift
        self.position_temperature = position_temperature
        self.class_temperature = class_temperature
        self.preprocess_prompt = preprocess_prompt
        self.postprocess_output = postprocess_output
        self.tail_padding_s = tail_padding_s
        self.audio_chunk_duration = audio_chunk_duration
        self.audio_chunk_threshold = audio_chunk_threshold

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

        gen_kwargs = {
            "num_step": self.num_step,
            "denoise": self.denoise,
            "guidance_scale": self.guidance_scale,
            "t_shift": self.t_shift,
            "position_temperature": self.position_temperature,
            "class_temperature": self.class_temperature,
            "preprocess_prompt": self.preprocess_prompt,
            "postprocess_output": self.postprocess_output,
            "audio_chunk_duration": self.audio_chunk_duration,
            "audio_chunk_threshold": self.audio_chunk_threshold,
        }

        def _synth(speed: float) -> np.ndarray:
            try:
                audios = self._model.generate(
                    text=text, language=lang, ref_audio=(wav, sr),
                    ref_text=ref_text, speed=float(speed),
                    **gen_kwargs
                )
            except Exception as exc:  # noqa: BLE001 - retry without conditioning
                LOG.warning(f"OmniVoice generate failed ({exc}); retrying minimally")
                # If we retry, we might want to still keep some settings, 
                # but typically retry is for "it failed because of specific conditioning"
                audios = self._model.generate(
                    text=text, language=lang, ref_audio=(wav, sr), ref_text=ref_text,
                    **gen_kwargs
                )
            if not audios:
                raise RuntimeError("OmniVoice returned no audio")
            a = audios[0]
            if isinstance(a, torch.Tensor):
                a = a.detach().cpu().numpy()
            samples = np.asarray(a, dtype=np.float32).reshape(-1)

            # Apply a small fade-out (50ms) to the end of generated speech 
            # to prevent clicks and abrupt cutoffs.
            fade_len = int(min(len(samples), 0.05 * self.sample_rate))
            if fade_len > 0:
                fade_curve = np.linspace(1.0, 0.0, fade_len)
                samples[-fade_len:] *= fade_curve

            # Apply tail padding to give the speech room to breathe 
            # (especially useful if OmniVoice trimmed it too tightly).
            if self.tail_padding_s > 0:
                pad_len = int(self.tail_padding_s * self.sample_rate)
                samples = np.concatenate([samples, np.zeros(pad_len, dtype=np.float32)])

            return samples

        seg = fit_by_speed(
            _synth, self.sample_rate, target_duration, base_speed=base_speed,
            tol=self.tolerance, max_speed=self.max_speed, max_iters=self.max_fit_iters,
        )
        seg.backend = self.backend_id
        return seg
