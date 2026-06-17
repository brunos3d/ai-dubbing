"""F5-TTS backend — an alternative flow-matching synthesizer.

F5-TTS (https://github.com/SWivid/F5-TTS) does zero-shot voice cloning from a
reference clip + its transcript, runs on CUDA, and exposes a ``speed`` control
we drive through the shared :func:`fit_by_speed` loop for duration targeting.

The ``f5_tts`` package is imported lazily inside :meth:`_load` so this backend
can be *selected* and constructed (and unit-tested via an injected model)
without the heavy dependency installed. Multilingual synthesis is supported by
pointing ``model``/``ckpt_file``/``vocab_file`` at a language-specific
checkpoint; the default is the base English/Chinese model.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..utils.logging import get_logger
from ..utils.vram import free_vram_aggressive
from .base import AudioSegment, SpeakerProfile, VoiceSynthesizer, fit_by_speed

LOG = get_logger("ai-dubbing.tts.f5tts")

_F5_SR = 24000


class F5TTSSynthesizer(VoiceSynthesizer):
    """F5-TTS voice-cloning backend."""

    backend_id = "f5tts"

    def __init__(
        self,
        *,
        language: Optional[str] = None,
        device: str = "cuda",
        model: str = "F5TTS_v1_Base",
        ckpt_file: str = "",
        vocab_file: str = "",
        tolerance: float = 0.10,
        max_speed: float = 1.35,
        max_fit_iters: int = 2,
        model_obj: Any = None,
    ) -> None:
        super().__init__(
            language=language, device=device, tolerance=tolerance,
            max_speed=max_speed, max_fit_iters=max_fit_iters,
        )
        self.model = model
        self.ckpt_file = ckpt_file
        self.vocab_file = vocab_file
        # ``model_obj`` lets tests inject a fake F5TTS without the dependency.
        self._model = model_obj
        self._model_sr = _F5_SR
        if model_obj is not None:
            self._loaded = True

    # -- lifecycle ---------------------------------------------------------

    def _load(self) -> None:
        try:
            from f5_tts.api import F5TTS
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "F5-TTS backend selected but the 'f5-tts' package is not "
                "importable. Install it with `pip install f5-tts`. "
                f"(import error: {exc})"
            ) from exc

        LOG.info(f"Loading F5-TTS ({self.model}) on {self.device}")
        kwargs: dict = {"model": self.model, "device": self.device}
        if self.ckpt_file:
            kwargs["ckpt_file"] = self.ckpt_file
        if self.vocab_file:
            kwargs["vocab_file"] = self.vocab_file
        try:
            self._model = F5TTS(**kwargs)
        except TypeError:
            # Older API used ``model_type`` instead of ``model``.
            kwargs.pop("model", None)
            self._model = F5TTS(model_type=self.model, device=self.device)
        self._model_sr = int(getattr(self._model, "target_sample_rate", _F5_SR) or _F5_SR)

    def _close(self) -> None:
        """Unload the model and aggressively free CUDA memory."""
        if self._model is not None:
            try:
                # Move model to CPU first to release GPU memory
                if hasattr(self._model, 'cpu'):
                    try:
                        self._model.cpu()
                    except Exception:
                        pass
                # Delete the model reference
                del self._model
            except Exception:
                pass
            self._model = None
        # Use aggressive cleanup to fully release CUDA memory
        free_vram_aggressive()

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
        ref_file = speaker_profile.reference_wav
        ref_text = speaker_profile.reference_text or ""

        def _synth(speed: float) -> np.ndarray:
            wav = self._infer(ref_file, ref_text, text, float(speed))
            return np.asarray(wav, dtype=np.float32).reshape(-1)

        seg = fit_by_speed(
            _synth, self.sample_rate, target_duration, base_speed=base_speed,
            tol=self.tolerance, max_speed=self.max_speed, max_iters=self.max_fit_iters,
        )
        seg.backend = self.backend_id
        return seg

    def _infer(self, ref_file: str, ref_text: str, gen_text: str, speed: float) -> np.ndarray:
        """Call F5TTS.infer, tolerating signature differences across versions."""
        try:
            out = self._model.infer(
                ref_file=ref_file,
                ref_text=ref_text,
                gen_text=gen_text,
                speed=speed,
                remove_silence=False,
                show_info=lambda *a, **k: None,
            )
        except TypeError:
            out = self._model.infer(ref_file, ref_text, gen_text, speed=speed)
        # F5TTS.infer returns (wav, sample_rate, spectrogram).
        if isinstance(out, (tuple, list)):
            wav = out[0]
            if len(out) > 1 and isinstance(out[1], int):
                self._model_sr = int(out[1])
        else:
            wav = out
        try:
            import torch
            if isinstance(wav, torch.Tensor):
                wav = wav.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(wav, dtype=np.float32).reshape(-1)
