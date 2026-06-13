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

from ..utils.entities import EntityPreserver
from ..utils.logging import get_logger, stage_banner
from ..utils.vram import free_vram, log_vram

LOG = get_logger("ai-dubbing.translate")


class NLLBTranslator:
    """Self-hosted translation using Meta's NLLB-200."""

    def __init__(self, model_id: str = "facebook/nllb-200-distilled-600M", device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.tokenizer = None

    def _load(self):
        if self.model is None:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            LOG.info(f"Loading NLLB model: {self.model_id} on {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            ).to(self.device)

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        self._load()
        import torch

        # Map simple codes to NLLB long-form codes
        # (Very simplified mapping, NLLB expects e.g. 'por_Latn' or 'eng_Latn')
        mapping = {
            "pt": "por_Latn",
            "en": "eng_Latn",
            "es": "spa_Latn",
            "fr": "fra_Latn",
            "de": "deu_Latn",
            "ru": "rus_Cyrl",
            "it": "ita_Latn",
            "ja": "jpn_Jpan",
            "zh": "zho_Hans",
        }

        src = mapping.get(src_lang, src_lang)
        tgt = mapping.get(tgt_lang, tgt_lang)

        # Use the modern src_lang/tgt_lang API on the tokenizer; fall back to
        # convert_tokens_to_ids for the forced BOS token (the legacy
        # ``lang_code_to_id`` attribute is not present on ``NllbTokenizer``).
        try:
            inputs = self.tokenizer(text, return_tensors="pt", src_lang=src, tgt_lang=tgt).to(self.device)
        except TypeError:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        bos_id = self.tokenizer.convert_tokens_to_ids(tgt)
        if bos_id is None or bos_id == self.tokenizer.unk_token_id:
            raise RuntimeError(f"NLLB tokenizer cannot resolve target language code: {tgt}")
        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=bos_id,
            max_length=256,
        )
        return self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]


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
    device: str = "cuda",
    glossary_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    src = _norm_lang(source)
    tgt = _norm_lang(target)
    if src == tgt:
        LOG.info("Source and target are the same; copying transcript")
        return [dict(s) for s in segments]

    # Initialize tools
    preserver = EntityPreserver(glossary_path)
    translator = NLLBTranslator(device=device)

    out: List[Dict[str, Any]] = []
    total = len(segments)
    nllb_ok = 0
    nllb_fail = 0

    for i, seg in enumerate(segments):
        text = (seg.get("text") or "").strip()
        if not text or seg.get("is_non_speech"):
            out.append(dict(seg))
            continue

        try:
            # 1. Protect Entities
            protected = preserver.protect(text)

            # 2. Translate (NLLB is the canonical backend; online is a
            #    last-resort safety net that is logged loudly so it cannot
            #    silently mask a real NLLB regression).
            translated = None
            used_nllb = False
            try:
                translated = translator.translate(protected, src, tgt)
                used_nllb = True
                nllb_ok += 1
            except Exception as nllb_exc:
                nllb_fail += 1
                LOG.error(
                    f"NLLB translation failed for segment {i}: {nllb_exc}. "
                    f"Falling back to online translator. This usually means a "
                    f"bug in the NLLB integration; please report it."
                )
                try:
                    translated = _safe_translate(protected, src, tgt)
                except Exception as online_exc:
                    LOG.error(
                        f"Online fallback also failed for segment {i}: {online_exc}. "
                        f"Keeping original text."
                    )
                    translated = protected

            # 3. Restore Entities
            final_text = preserver.restore(translated)

            # 4. Duration Awareness (Log warnings for now)
            src_words = len(text.split())
            tgt_words = len(final_text.split())
            if src_words > 0 and (tgt_words / src_words > 2.0 or tgt_words / src_words < 0.5):
                LOG.warning(f"Segment {i+1}: Translation length anomaly ({src_words} -> {tgt_words} words)")

        except Exception as exc:  # noqa: BLE001
            LOG.error(f"Translation failed for segment {i}: {exc}")
            final_text = text

        new = dict(seg)
        new["source_text"] = text
        new["text"] = final_text
        new["entities_preserved"] = list(preserver.placeholders.values())
        new["translation_backend"] = "nllb" if used_nllb else "fallback"
        out.append(new)

        if (i + 1) % 10 == 0 or i == total - 1:
            LOG.info(f"Translated {i + 1}/{total} segments")

    if total:
        LOG.info(
            f"Translation summary: NLLB={nllb_ok}/{total}, "
            f"fallback={nllb_fail}/{total}"
        )

    return out


class TranslateStage:
    name = "translate"

    def __init__(
        self,
        workdir: Path,
        source_language: str,
        target_language: str,
        device: str = "cuda",
        glossary_path: Optional[Path] = None,
    ):
        self.workdir = Path(workdir)
        self.source_language = source_language
        self.target_language = target_language
        self.device = device
        self.glossary_path = glossary_path

    def outputs(self) -> List[Path]:
        return [self.workdir / "translated_transcript.json"]

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        stage_banner(LOG, 5, 12, "Translation")
        transcript_path = Path(context["transcript_path"])
        out_path = self.workdir / "translated_transcript.json"
        
        # Look for glossary in PROJECT_ROOT if not explicitly provided
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
            device=self.device,
            glossary_path=g_path
        )
        
        out_path.write_text(json.dumps(translated, indent=2, ensure_ascii=False))
        LOG.info(f"Translated transcript -> {out_path} ({len(translated)} segments)")
        log_vram(LOG)
        free_vram()
        return {"translated_path": str(out_path), "num_segments": len(translated)}
