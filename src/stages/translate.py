"""Stage 6 - Translation of transcripts to the target language.

Strategy:
1. Try deep-translator (GoogleTranslator / MyMemory) for free online translation.
2. If all providers fail, raise a clear error so the user can install offline models.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.logging import get_logger, stage_banner
from ..utils.vram import free_vram, log_vram

LOG = get_logger("ai-dubbing.translate")


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


def _make_translator(provider: str = "google"):
    from deep_translator import GoogleTranslator, MyMemoryTranslator

    if provider == "google":
        return GoogleTranslator
    if provider == "mymemory":
        return MyMemoryTranslator
    raise ValueError(f"Unknown provider: {provider}")


def _safe_translate(text: str, source: str, target: str) -> str:
    from deep_translator import GoogleTranslator, MyMemoryTranslator, exceptions

    last_err: Optional[Exception] = None
    for provider in ("google", "mymemory"):
        try:
            if provider == "google":
                translator = GoogleTranslator(source=source, target=target)
            else:
                translator = MyMemoryTranslator(source=source, target=target)
            out = translator.translate(text)
            if out:
                return out
        except exceptions.NotValidPayload as e:
            LOG.warning(f"{provider}: empty payload ({e})")
            last_err = e
        except exceptions.TranslationNotFound as e:
            LOG.warning(f"{provider}: not found ({e})")
            last_err = e
        except Exception as e:  # noqa: BLE001
            LOG.warning(f"{provider}: error ({e})")
            last_err = e
            time.sleep(0.4)
    raise RuntimeError(f"All translators failed: {last_err}")


def translate_segments(
    segments: List[Dict[str, Any]],
    source: str,
    target: str,
    sleep_seconds: float = 0.05,
) -> List[Dict[str, Any]]:
    src = _norm_lang(source)
    tgt = _norm_lang(target)
    if src == tgt:
        LOG.info("Source and target are the same; copying transcript")
        return [dict(s) for s in segments]

    out: List[Dict[str, Any]] = []
    total = len(segments)
    for i, seg in enumerate(segments):
        text = (seg.get("text") or "").strip()
        if not text or seg.get("is_non_speech"):
            if seg.get("is_non_speech"):
                LOG.info(f"Segment {i + 1}: skipping translation for non-speech event: {text}")
            out.append(dict(seg))
            continue
        try:
            translated = _safe_translate(text, src, tgt)
        except Exception as exc:  # noqa: BLE001
            LOG.error(f"Translation failed for segment {i}: {exc}")
            translated = text
        new = dict(seg)
        new["source_text"] = text
        new["text"] = translated
        out.append(new)
        if (i + 1) % 5 == 0 or i == total - 1:
            LOG.info(f"Translated {i + 1}/{total} segments")
        time.sleep(sleep_seconds)
    return out


class TranslateStage:
    name = "translate"

    def __init__(
        self,
        workdir: Path,
        source_language: str,
        target_language: str,
    ):
        self.workdir = Path(workdir)
        self.source_language = source_language
        self.target_language = target_language

    def outputs(self) -> List[Path]:
        return [self.workdir / "translated_transcript.json"]

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        stage_banner(LOG, 5, 11, "Translation")
        transcript_path = Path(context["transcript_path"])
        out_path = self.workdir / "translated_transcript.json"
        segments = json.loads(transcript_path.read_text(encoding="utf-8"))
        translated = translate_segments(segments, self.source_language, self.target_language)
        out_path.write_text(json.dumps(translated, indent=2, ensure_ascii=False))
        LOG.info(f"Translated transcript -> {out_path} ({len(translated)} segments)")
        log_vram(LOG)
        free_vram()
        return {"translated_path": str(out_path), "num_segments": len(translated)}
