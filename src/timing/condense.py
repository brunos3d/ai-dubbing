"""Deterministic translation condensation for timing fit (spec §2, Failure #3).

The timing selector can only prefer a better-fitting translation if it is
given more than one candidate.  Online translators return a single literal
rendering per provider, and Portuguese is markedly longer than English, so
every line overflowed its slot.

This module produces *shorter, still-natural* candidates with pure string
rules — no network, no LLM — so the selector (``timing.score``) has a real
choice.  The rules are conservative and language-gated; the fidelity term in
the scorer penalises any candidate that cut too much, so an over-aggressive
condensation simply loses to the literal one.

Portuguese levers (both idiomatic):

* **pro-drop** — Portuguese routinely omits subject pronouns because the verb
  already encodes person/number ("Você está" → "Está"), saving a whole word;
* **filler trimming** — leading discourse markers ("Bem,", "Quero dizer,")
  carry no propositional content and are dropped.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

# Subject pronouns that Portuguese can drop when a verb follows (pro-drop).
_PT_SUBJECT_PRONOUNS = {
    "eu", "você", "voce", "ele", "ela", "nós", "nos", "eles", "elas", "vocês", "voces",
}

# Leading discourse fillers (matched case-insensitively, comma-terminated).
_PT_LEADING_FILLERS = [
    "bem", "quero dizer", "quer dizer", "olha", "veja", "sabe", "então", "entao",
    "na verdade", "tipo", "ok", "tudo bem", "ah", "é", "eh", "digo",
]


def _norm_lang(code: Optional[str]) -> str:
    code = (code or "").strip().lower()
    return code.split("-")[0].split("_")[0]


def _strip_leading_fillers(text: str) -> str:
    changed = True
    while changed:
        changed = False
        for f in _PT_LEADING_FILLERS:
            # "Filler, rest" -> "rest"  (filler followed by a comma)
            m = re.match(rf"^\s*{re.escape(f)}\s*,\s*(.+)$", text, flags=re.IGNORECASE)
            if m and m.group(1).strip():
                rest = m.group(1).strip()
                # Recapitalise so the trimmed sentence still reads naturally.
                text = rest[0].upper() + rest[1:] if rest else rest
                changed = True
                break
    return text


def _drop_subject_pronoun(text: str) -> str:
    """Drop a leading subject pronoun when a (verb) word follows it."""
    m = re.match(r"^\s*([A-Za-zÀ-ÿ]+)\s+([A-Za-zÀ-ÿ].*)$", text)
    if not m:
        return text
    first, rest = m.group(1), m.group(2)
    if first.lower() in _PT_SUBJECT_PRONOUNS:
        # Capitalise the new first word so the sentence still reads naturally.
        rest = rest[0].upper() + rest[1:] if rest else rest
        return rest
    return text


def condense(text: str, lang: Optional[str]) -> Optional[str]:
    """Return a shorter, natural variant of ``text`` or ``None`` if unchanged.

    Only Portuguese is handled today; other languages return ``None`` so the
    caller keeps just the literal candidate.
    """
    if _norm_lang(lang) != "pt" or not text or not text.strip():
        return None

    out = _strip_leading_fillers(text.strip())
    out = _drop_subject_pronoun(out)
    out = re.sub(r"\s+", " ", out).strip()

    if out and out != text.strip() and len(out) < len(text.strip()):
        return out
    return None


def condense_candidates(
    text: str, backend: str, lang: Optional[str]
) -> List[Tuple[str, str]]:
    """Return ``[(original, backend)]`` plus a condensed variant when possible.

    The condensed candidate's backend label is suffixed with ``+condensed`` so
    the timing report records which selections came from condensation.
    """
    out: List[Tuple[str, str]] = [(text, backend)]
    short = condense(text, lang)
    if short:
        out.append((short, f"{backend}+condensed"))
    return out
