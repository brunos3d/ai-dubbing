"""Stage 6 - Translation of transcripts to the target language.

Architecture
------------
Translation is delegated to a pluggable :class:`TranslationBackend`.  The
pipeline (entity-preservation, length diagnostics, video-stitching) only
knows about the abstract interface; swapping a backend never touches
pipeline internals.

Currently registered backends:

* :class:`DeepTranslatorBackend` - the stable default.  Uses the
  ``deep-translator`` package with two online providers (Google, then
  MyMemory as a fallback) and rate-limits itself.  This is the same
  implementation that has shipped since the initial commit; it produces
  literal but stable translations that round-trip named entities
  correctly and never hallucinates legal text.

Adding a new backend is just a matter of subclassing
:class:`TranslationBackend` and registering it in ``BACKEND_REGISTRY``.
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.entities import EntityPreserver
from ..utils.logging import get_logger, stage_banner
from ..utils.vram import free_vram, log_vram

LOG = get_logger("ai-dubbing.translate")


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------


class TranslationBackend(ABC):
    """Pluggable translation backend interface.

    Subclasses must implement :meth:`translate` and expose a stable
    :attr:`name` so the pipeline can record which backend produced a
    given segment (useful for debugging and for picking the best
    backend in production).
    """

    name: str = "abstract"

    @abstractmethod
    def translate(self, text: str, source: str, target: str) -> str:
        """Translate ``text`` from ISO-639-1 ``source`` to ``target``."""


class DeepTranslatorBackend(TranslationBackend):
    """Stable online translator (Google -> MyMemory fallback).

    This is the canonical backend restored after the NLLB migration was
    rolled back.  It is the same implementation that powered the
    pipeline before NLLB: it produces literal but well-formed output,
    handles named entities conservatively, and never hallucinates
    legal/regulatory text.
    """

    name = "deep-translator"

    def __init__(
        self,
        providers: Optional[List[str]] = None,
        sleep_seconds: float = 0.05,
        per_provider_sleep: float = 0.4,
    ):
        # Default provider order: Google (best quality), then MyMemory.
        self.providers = providers or ["google", "mymemory"]
        self.sleep_seconds = sleep_seconds
        self.per_provider_sleep = per_provider_sleep

    def _make_translator(self, provider: str, source: str, target: str):
        from deep_translator import GoogleTranslator, MyMemoryTranslator

        if provider == "google":
            return GoogleTranslator(source=source, target=target)
        if provider == "mymemory":
            return MyMemoryTranslator(source=source, target=target)
        raise ValueError(f"Unknown provider: {provider}")

    def translate(self, text: str, source: str, target: str) -> str:
        from deep_translator import exceptions

        last_err: Optional[Exception] = None
        for provider in self.providers:
            try:
                translator = self._make_translator(provider, source, target)
                out = translator.translate(text)
                if out:
                    return out
            except exceptions.NotValidPayload as exc:
                LOG.warning(f"{provider}: empty payload ({exc})")
                last_err = exc
            except exceptions.TranslationNotFound as exc:
                LOG.warning(f"{provider}: not found ({exc})")
                last_err = exc
            except Exception as exc:  # noqa: BLE001
                LOG.warning(f"{provider}: error ({exc})")
                last_err = exc
                time.sleep(self.per_provider_sleep)
        raise RuntimeError(
            f"All {self.name} providers {self.providers!r} failed: {last_err}"
        )


class MarianBackend(TranslationBackend):
    """Stub for a future self-hosted MarianMT backend.

    Kept here as a documented example of how to plug in a new
    self-hosted backend without touching pipeline internals.  The
    class is intentionally not wired into :data:`BACKEND_REGISTRY` so
    the default behaviour stays stable; instantiate it directly to
    experiment.
    """

    name = "marian"

    def translate(self, text: str, source: str, target: str) -> str:
        raise NotImplementedError(
            "MarianBackend is a documented extension point. "
            "Load a Helsinki-NLP/Opus model and route the call here."
        )


class WhisperTranslationBackend(TranslationBackend):
    """Stub for a future Whisper-style translation backend.

    Whisper itself does not translate between arbitrary language pairs
    on its own, so a production implementation would chain
    transcription (source -> English) with a small LM.  Documented
    here as the integration shape; not active by default.
    """

    name = "whisper"

    def translate(self, text: str, source: str, target: str) -> str:
        raise NotImplementedError(
            "WhisperTranslationBackend is a documented extension point."
        )


# Registry of production-ready backends.  The first entry is the
# pipeline default.  Stubs are kept outside the registry so they
# cannot be selected by accident.
BACKEND_REGISTRY: Dict[str, type] = {
    "deep-translator": DeepTranslatorBackend,
}


def make_backend(name: Optional[str] = None, **kwargs) -> TranslationBackend:
    """Resolve a backend by name (defaults to the first registered one)."""
    if name is None:
        name = next(iter(BACKEND_REGISTRY))
    if name not in BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown translation backend: {name!r}. "
            f"Available: {sorted(BACKEND_REGISTRY)}"
        )
    return BACKEND_REGISTRY[name](**kwargs)


# ---------------------------------------------------------------------------
# Language normalisation
# ---------------------------------------------------------------------------


def _norm_lang(code: str) -> str:
    code = (code or "").strip().lower()
    if code in {"pt", "pt-br", "pt_br"}:
        return "pt"
    if code in {"en", "en-us", "en-gb"}:
        return "en"
    if code in {"es", "es-es", "es-mx"}:
        return "es"
    if "-" in code:
        return code.split("-")[0]
    return code


# ---------------------------------------------------------------------------
# Public pipeline-facing API
# ---------------------------------------------------------------------------


def translate_segments(
    segments: List[Dict[str, Any]],
    source: str,
    target: str,
    glossary_path: Optional[Path] = None,
    backend: Optional[TranslationBackend] = None,
) -> List[Dict[str, Any]]:
    """Translate every segment, preserving entities and recording backend.

    Pipeline callers can pass a pre-built ``backend``; otherwise the
    default :class:`DeepTranslatorBackend` is instantiated.
    """
    src = _norm_lang(source)
    tgt = _norm_lang(target)
    if src == tgt:
        LOG.info("Source and target are the same; copying transcript")
        return [dict(s) for s in segments]

    backend = backend or make_backend()
    LOG.info(f"Using translation backend: {backend.name}")

    preserver = EntityPreserver(glossary_path)
    if preserver.is_active:
        LOG.info(f"Entity preservation active for {preserver._loaded_count} glossary terms")
    else:
        LOG.info("No glossary supplied; entity preservation disabled, "
                 "translator will see the source text verbatim")
    out: List[Dict[str, Any]] = []
    total = len(segments)
    backend_ok = 0
    backend_fail = 0

    for i, seg in enumerate(segments):
        text = (seg.get("text") or "").strip()
        is_non_speech = bool(seg.get("is_non_speech"))
        if not text or is_non_speech:
            if is_non_speech:
                LOG.debug(f"Segment {i + 1}: keeping non-speech event ({text!r})")
            out.append(dict(seg))
            continue

        # 1. Protect entities ONLY if a glossary was supplied. When the
        #    glossary is empty, the preserver is a pass-through and the
        #    translator sees the original text verbatim.
        protected = preserver.protect(text)

        # 2. Translate through the active backend.
        translated: Optional[str] = None
        used_backend = False
        try:
            translated = backend.translate(protected, src, tgt)
            used_backend = True
            backend_ok += 1
        except Exception as backend_exc:
            backend_fail += 1
            LOG.error(
                f"Backend {backend.name!r} failed for segment {i}: {backend_exc}. "
                f"Keeping original text for this segment; the pipeline will "
                f"continue so a single bad segment does not abort the run."
            )
            translated = protected  # fall back to the protected original

        # 3. Restore entities. No-op when no glossary was supplied.
        final_text = preserver.restore(translated)

        # 4. Length-diagnostics: warn on suspicious length ratios so a
        #    backend regression is obvious in the logs without failing
        #    the run.
        src_words = len(text.split())
        tgt_words = len(final_text.split())
        if src_words > 0:
            ratio = tgt_words / src_words
            if ratio > 2.0 or ratio < 0.5:
                LOG.warning(
                    f"Segment {i + 1}: translation length anomaly "
                    f"({src_words} -> {tgt_words} words, ratio {ratio:.2f})"
                )

        new = dict(seg)
        new["source_text"] = text
        new["text"] = final_text
        if preserver.placeholders:
            new["entities_preserved"] = list(preserver.placeholders.values())
        new["translation_backend"] = backend.name if used_backend else "passthrough"
        out.append(new)

        if (i + 1) % 10 == 0 or i == total - 1:
            LOG.info(f"Translated {i + 1}/{total} segments")

    if total:
        LOG.info(
            f"Translation summary: backend={backend.name} "
            f"ok={backend_ok}/{total} passthrough={backend_fail}/{total}"
        )

    return out


class TranslateStage:
    name = "translate"

    def __init__(
        self,
        workdir: Path,
        source_language: str,
        target_language: str,
        glossary_path: Optional[Path] = None,
        backend: Optional[TranslationBackend] = None,
        backend_name: Optional[str] = None,
    ):
        self.workdir = Path(workdir)
        self.source_language = source_language
        self.target_language = target_language
        self.glossary_path = glossary_path
        # ``backend`` is honoured if the caller already built one; otherwise
        # we resolve ``backend_name`` (default: the registered default).
        self.backend = backend or (
            make_backend(backend_name) if backend_name else make_backend()
        )

    def outputs(self) -> List[Path]:
        return [self.workdir / "translated_transcript.json"]

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        stage_banner(LOG, 5, 11, "Translation")
        transcript_path = Path(context["transcript_path"])
        out_path = self.workdir / "translated_transcript.json"

        # Look for glossary in PROJECT_ROOT if not explicitly provided.
        g_path = self.glossary_path
        if not g_path:
            from ..utils.paths import project_root
            cand = project_root() / "entity_glossary.json"
            if cand.exists():
                g_path = cand

        segments = json.loads(transcript_path.read_text(encoding="utf-8"))
        translated = translate_segments(
            segments,
            self.source_language,
            self.target_language,
            glossary_path=g_path,
            backend=self.backend,
        )

        out_path.write_text(json.dumps(translated, indent=2, ensure_ascii=False))
        LOG.info(f"Translated transcript -> {out_path} ({len(translated)} segments)")
        log_vram(LOG)
        free_vram()
        return {"translated_path": str(out_path), "num_segments": len(translated)}
