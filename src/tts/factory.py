"""Backend registry and factory for :class:`VoiceSynthesizer`.

The pipeline never imports a concrete backend; it calls :func:`make_synthesizer`
with a backend id (persisted in workspace metadata). Adding a backend is one
registry entry.
"""
from __future__ import annotations

from typing import Dict, List, Type

from .base import VoiceSynthesizer
from .f5tts import F5TTSSynthesizer
from .omnivoice import OmniVoiceSynthesizer

#: Registry of backend id -> class. The first entry is the default.
BACKENDS: Dict[str, Type[VoiceSynthesizer]] = {
    OmniVoiceSynthesizer.backend_id: OmniVoiceSynthesizer,
    F5TTSSynthesizer.backend_id: F5TTSSynthesizer,
}

DEFAULT_BACKEND = OmniVoiceSynthesizer.backend_id


def available_backends() -> List[str]:
    """Backend ids that can be selected (default first)."""
    return list(BACKENDS.keys())


def is_known_backend(name: str) -> bool:
    return (name or "").strip().lower() in BACKENDS


def make_synthesizer(backend: str | None = None, **kwargs) -> VoiceSynthesizer:
    """Construct a synthesizer for ``backend`` (defaults to OmniVoice).

    Extra kwargs (language, device, fitting bounds, model overrides) are passed
    to the backend constructor. Construction is cheap — the model loads lazily
    on first :meth:`~VoiceSynthesizer.generate`.
    """
    name = (backend or DEFAULT_BACKEND).strip().lower()
    if name not in BACKENDS:
        raise ValueError(
            f"Unknown TTS backend: {backend!r}. Available: {available_backends()}"
        )
    return BACKENDS[name](**kwargs)
