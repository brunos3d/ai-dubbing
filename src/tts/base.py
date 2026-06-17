"""The :class:`VoiceSynthesizer` abstraction and its data types.

Every TTS backend implements :class:`VoiceSynthesizer`. The pipeline passes a
:class:`SpeakerProfile` (built from the existing speaker artifacts — it does
*not* redesign speaker extraction) and an optional ``target_duration``; the
backend returns an :class:`AudioSegment`.

Duration fitting is shared, not backend-specific: :func:`fit_by_speed` runs the
generic "synthesize, measure, speed up if it overruns" loop so every backend
gets timing-as-a-constraint behaviour from one place. Underruns are accepted
as natural pauses (consistent with the asymmetric align stage).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

_DEFAULT_SR = 24000


@dataclass
class SpeakerProfile:
    """A voice-cloning reference, sourced from the existing samples stage.

    ``reference_wav`` is the path to ``primary/reference.wav`` and
    ``reference_text`` its transcript (the clone prompt). Nothing here is new —
    it is a typed view over artifacts the pipeline already produces.
    """

    speaker_id: str
    reference_wav: str
    reference_text: Optional[str] = None


@dataclass
class AudioSegment:
    """Synthesized audio plus the metadata the pipeline records per segment."""

    samples: np.ndarray
    sample_rate: int
    duration_s: float
    speed: float = 1.0
    iterations: int = 1
    backend: str = ""

    @property
    def num_samples(self) -> int:
        return int(self.samples.shape[-1]) if self.samples is not None else 0


def next_speed(
    measured: float,
    slot: float,
    cur_speed: float,
    max_speed: float,
    tol: float,
) -> Optional[float]:
    """Next speaking speed to try, or ``None`` when no further pass helps.

    Only overruns (speech longer than the slot, which bleed into the next turn)
    are corrected, by scaling speed up proportionally and clamping to
    ``max_speed`` for naturalness. Underruns are left as natural pauses.
    """
    if slot <= 0 or measured <= 0:
        return None
    if measured <= slot * (1.0 + tol):
        return None
    target = min(cur_speed * (measured / slot), max_speed)
    if target <= cur_speed + 1e-3:
        return None
    return round(target, 3)


def fit_by_speed(
    synth_fn: Callable[[float], np.ndarray],
    sample_rate: int,
    slot_s: Optional[float],
    base_speed: float = 1.0,
    tol: float = 0.10,
    max_speed: float = 1.35,
    max_iters: int = 2,
) -> AudioSegment:
    """Generic duration-fitting loop usable by any backend.

    ``synth_fn(speed) -> np.ndarray`` synthesizes at a given speed. We measure
    the result and, only for overruns, raise the speed and regenerate (bounded
    by ``max_speed`` over ``max_iters`` passes), returning the least-overrun
    candidate. When ``slot_s`` is falsy we synthesize once at ``base_speed``.
    """
    speed = float(base_speed)
    audio = np.asarray(synth_fn(speed), dtype=np.float32).reshape(-1)
    dur = audio.shape[0] / sample_rate if sample_rate else 0.0
    iters = 1
    if not slot_s or slot_s <= 0:
        return AudioSegment(audio, sample_rate, dur, speed, iters)

    best = (audio, speed, dur)
    for _ in range(max_iters):
        ns = next_speed(dur, slot_s, speed, max_speed, tol)
        if ns is None:
            break
        speed = ns
        audio = np.asarray(synth_fn(speed), dtype=np.float32).reshape(-1)
        dur = audio.shape[0] / sample_rate if sample_rate else 0.0
        iters += 1
        if max(0.0, dur - slot_s) < max(0.0, best[2] - slot_s):
            best = (audio, speed, dur)
    if max(0.0, dur - slot_s) > max(0.0, best[2] - slot_s):
        audio, speed, dur = best
    return AudioSegment(audio, sample_rate, dur, speed, iters)


# Shared ISO-code -> spoken language name table (OmniVoice & F5 both want names).
LANGUAGE_NAME = {
    "en": "English",
    "pt": "Portuguese",
    "pt-br": "Portuguese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ru": "Russian",
}


def language_name(code: Optional[str]) -> str:
    code = (code or "auto").lower()
    if code in LANGUAGE_NAME:
        return LANGUAGE_NAME[code]
    if "-" in code and code.split("-")[0] in LANGUAGE_NAME:
        return LANGUAGE_NAME[code.split("-")[0]]
    return code


class VoiceSynthesizer(ABC):
    """Common interface every TTS backend implements.

    Lifecycle: construct (cheap) -> :meth:`load` (lazy, loads the model) ->
    :meth:`generate` per segment -> :meth:`close` (free GPU memory). ``load``
    and ``close`` are idempotent.
    """

    #: Stable identifier persisted in workspace metadata and the manifest.
    backend_id: str = "abstract"

    def __init__(
        self,
        *,
        language: Optional[str] = None,
        device: str = "cuda",
        tolerance: float = 0.10,
        max_speed: float = 1.35,
        max_fit_iters: int = 2,
    ) -> None:
        self.language = language
        self.device = device
        self.tolerance = tolerance
        self.max_speed = max_speed
        self.max_fit_iters = max_fit_iters
        self._loaded = False

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        """Load the model once. Idempotent; subclasses override :meth:`_load`."""
        if self._loaded:
            return
        self._load()
        self._loaded = True

    def _load(self) -> None:  # pragma: no cover - trivial default
        """Backend-specific model load. Default: nothing to load."""
        return None

    def close(self) -> None:
        """Release resources. Idempotent."""
        if not self._loaded:
            return
        self._close()
        self._loaded = False

    def _close(self) -> None:  # pragma: no cover - trivial default
        return None

    # -- properties --------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        return _DEFAULT_SR

    # -- synthesis ---------------------------------------------------------

    @abstractmethod
    def generate(
        self,
        text: str,
        speaker_profile: SpeakerProfile,
        target_duration: Optional[float] = None,
        *,
        base_speed: float = 1.0,
    ) -> AudioSegment:
        """Synthesize ``text`` in the speaker's cloned voice.

        When ``target_duration`` is given the backend should make the output
        fit the slot (typically via :func:`fit_by_speed`). Returns an
        :class:`AudioSegment` tagged with this backend's id.
        """

    # context-manager sugar
    def __enter__(self) -> "VoiceSynthesizer":
        self.load()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
