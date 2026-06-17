"""Pluggable text-to-speech backends.

The dubbing pipeline depends only on the :class:`VoiceSynthesizer` abstraction
(``base.py``) and obtains a concrete backend from :func:`make_synthesizer`
(``factory.py``). OmniVoice remains the default; F5-TTS is an alternative.
Adding a backend is a matter of subclassing :class:`VoiceSynthesizer` and
registering it in the factory.
"""
from .base import AudioSegment, SpeakerProfile, VoiceSynthesizer
from .factory import (
    BACKENDS,
    available_backends,
    is_known_backend,
    make_synthesizer,
)

__all__ = [
    "AudioSegment",
    "SpeakerProfile",
    "VoiceSynthesizer",
    "make_synthesizer",
    "available_backends",
    "is_known_backend",
    "BACKENDS",
]
