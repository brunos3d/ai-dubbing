"""Language-aware spoken-duration estimation (spec 2026-06-16 §2.1).

The core idea behind timing-aware dubbing is to know *how long a candidate
translation will take to speak* **before** synthesizing it, so the pipeline
can pick a translation that fits the slot rather than stretching audio after
the fact.

Model: ``duration ≈ syllables / speaking_rate + pause_budget``.

* **Syllables** are counted with a vowel-group heuristic for alphabetic
  scripts, and a character-count proxy for the syllable-/mora-dense CJK
  scripts where text-based vowel counting is meaningless.
* **Speaking rate** (syllables per second) is a per-language constant drawn
  from speech-tempo literature; it is overridable.
* **Pause budget** adds a small, capped amount of time for sentence- and
  clause-final punctuation, which real speech honours.

Everything here is a pure function — no torch, no I/O, no network — so it is
cheap and deterministic.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

# Speaking rate in *syllables per second*, by ISO-639-1 language code.
# Sources: cross-linguistic speech-tempo studies (Pellegrino et al. 2011 and
# follow-ups). These are deliberately conservative central estimates; the
# absolute values matter less than the *relative* ordering and consistency,
# because the score compares candidates within a single slot.
SPEAKING_RATE_SYL_S: dict[str, float] = {
    "en": 4.4,
    "es": 5.4,
    "pt": 4.9,
    "fr": 5.0,
    "it": 5.0,
    "de": 4.3,
    "nl": 4.4,
    "ru": 4.6,
    "ja": 5.7,   # mora-timed; with char-proxy syllables this rate fits
    "ko": 5.0,
    "zh": 5.0,   # syllable == hanzi
}
DEFAULT_RATE_SYL_S: float = 4.6

# Calibration: neural TTS (OmniVoice) speaks faster than the conversational
# reference tempos above. Measured on the Elon/Colbert workspace the literature
# rates over-predicted actual synthesized duration by a median of ~27% across
# 24 segments, which inflated every rate factor and the "needs stretch" count.
# Scaling the language-default rate up by this factor calibrates the estimate
# to the synthesis engine so the timing report and stretch decisions reflect
# reality. Applied only to language defaults; an explicit ``rate_syl_s``
# override (e.g. a speaker's measured tempo) is respected verbatim.
SYNTHESIS_TEMPO_FACTOR: float = 1.25

# Pause budget (seconds) added per punctuation mark.
_SENTENCE_PAUSE_S: float = 0.25
_CLAUSE_PAUSE_S: float = 0.12
# Cap total pause budget at this fraction of the raw speech time so a
# punctuation-heavy line can't be estimated as mostly silence.
_MAX_PAUSE_FRACTION: float = 0.30

_SENTENCE_PUNCT = ".?!…。！？"
_CLAUSE_PUNCT = ",;:、；："

_VOWELS = set("aeiouyàáâãäåèéêëìíîïòóôõöùúûüýÿæœ")

# Unicode ranges treated as "one syllable per character" (CJK + kana + hangul).
_CJK_RANGES = (
    (0x3040, 0x30FF),   # hiragana + katakana
    (0x3400, 0x4DBF),   # CJK ext A
    (0x4E00, 0x9FFF),   # CJK unified
    (0xAC00, 0xD7A3),   # hangul syllables
    (0xF900, 0xFAFF),   # CJK compat
)


def _norm_lang(code: Optional[str]) -> str:
    code = (code or "").strip().lower()
    if "-" in code:
        code = code.split("-")[0]
    if "_" in code:
        code = code.split("_")[0]
    return code


def _is_cjk_char(ch: str) -> bool:
    o = ord(ch)
    return any(lo <= o <= hi for lo, hi in _CJK_RANGES)


def _count_cjk_syllables(text: str) -> int:
    """One syllable per CJK/kana/hangul character; ignore everything else."""
    return sum(1 for ch in text if _is_cjk_char(ch))


def _strip_accents_lower(word: str) -> str:
    return word.lower()


def _syllables_in_word(word: str, lang: str) -> int:
    """Vowel-group syllable count for one alphabetic word (>= 1 if it has letters)."""
    w = _strip_accents_lower(word)
    letters = [c for c in w if c.isalpha()]
    if not letters:
        return 0
    groups = 0
    prev_vowel = False
    for c in w:
        is_vowel = c in _VOWELS
        if is_vowel and not prev_vowel:
            groups += 1
        prev_vowel = is_vowel

    # English: a single silent trailing 'e' usually adds no syllable
    # ("make", "time"), but keep "-le" endings ("table") and never go to 0.
    if lang == "en" and groups > 1 and w.endswith("e") and not w.endswith("le"):
        groups -= 1

    return max(1, groups)


def estimate_syllables(text: str, lang: Optional[str] = None) -> int:
    """Estimate the syllable count of ``text`` in language ``lang``."""
    if not text or not text.strip():
        return 0
    code = _norm_lang(lang)

    if code in {"zh", "ja"}:
        cjk = _count_cjk_syllables(text)
        # Any romaji / latin remainder is counted with the vowel heuristic.
        latin = "".join(ch if not _is_cjk_char(ch) else " " for ch in text)
        latin_syl = sum(
            _syllables_in_word(w, "en") for w in re.findall(r"[A-Za-z]+", latin)
        )
        return cjk + latin_syl

    # ``ko`` hangul blocks are one syllable each and fall in the CJK range.
    cjk = _count_cjk_syllables(text) if code == "ko" else 0
    words = re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)
    syl = sum(_syllables_in_word(w, code) for w in words)
    return syl + cjk


@dataclass(frozen=True)
class DurationEstimate:
    """The estimated spoken duration of a piece of text and its components."""

    seconds: float
    syllables: int
    rate_syl_s: float
    pause_s: float
    method: str

    def __float__(self) -> float:  # convenience: ``float(estimate)``
        return self.seconds


def speaking_rate(lang: Optional[str]) -> float:
    """Return the syllables-per-second rate for ``lang`` (or the default)."""
    return SPEAKING_RATE_SYL_S.get(_norm_lang(lang), DEFAULT_RATE_SYL_S)


def _pause_budget(text: str, speech_time_s: float) -> float:
    sentences = sum(text.count(p) for p in _SENTENCE_PUNCT)
    clauses = sum(text.count(p) for p in _CLAUSE_PUNCT)
    raw = sentences * _SENTENCE_PAUSE_S + clauses * _CLAUSE_PAUSE_S
    return min(raw, _MAX_PAUSE_FRACTION * speech_time_s) if speech_time_s > 0 else min(raw, 1.0)


def estimate_duration(
    text: str,
    lang: Optional[str] = None,
    *,
    rate_syl_s: Optional[float] = None,
) -> DurationEstimate:
    """Estimate how long ``text`` takes to speak naturally in ``lang``.

    ``rate_syl_s`` overrides the per-language speaking rate (useful for
    prosody-aware adjustment in Phase 5, where a speaker's measured tempo
    can be substituted).
    """
    if not text or not text.strip():
        return DurationEstimate(0.0, 0, rate_syl_s or speaking_rate(lang), 0.0, "empty")

    syllables = estimate_syllables(text, lang)
    if rate_syl_s and rate_syl_s > 0:
        rate = rate_syl_s
    else:
        rate = speaking_rate(lang) * SYNTHESIS_TEMPO_FACTOR
    speech_time = syllables / rate if rate > 0 else 0.0
    pause = _pause_budget(text, speech_time)
    total = speech_time + pause
    return DurationEstimate(
        seconds=round(total, 4),
        syllables=syllables,
        rate_syl_s=rate,
        pause_s=round(pause, 4),
        method="syllable+pause",
    )
