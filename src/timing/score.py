"""Translation-candidate timing score & selection (spec 2026-06-16 §2.3).

Given several translation candidates for one segment and the *slot* duration
(the original line's start→end span), pick the candidate that best balances:

* **duration fit** — how close its estimated spoken duration is to the slot;
* **rate naturalness** — how far the implied speaking-rate factor strays from
  the natural band (a fit achieved only by speaking unnaturally fast/slow is
  penalized);
* **fidelity** — a length-ratio proxy for "did the translation keep the
  content" (very short candidates likely dropped meaning; very long ones pad);
* **glossary compliance** — whether required glossary terms survived.

The point of the exercise (Phase 2.4) is to **prefer a better-fitting
translation over stretching audio**. The selected candidate's estimate is
usually within the natural band, so the downstream ``align`` stage rarely has
to time-stretch at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from .duration import estimate_duration

# Score weights (sum to 1.0). Single source of truth so they're tunable.
W_DURATION_FIT = 0.45
W_RATE = 0.20
W_FIDELITY = 0.20
W_GLOSSARY = 0.15

# Natural speaking-rate band: a rate factor (estimated / slot) inside this
# range is "free"; outside it is penalized proportionally.
NATURAL_RATE_LOW = 0.90
NATURAL_RATE_HIGH = 1.15
# Rate factor distance beyond the band at which the penalty saturates to 1.0.
_RATE_SATURATION = 0.60

# Fidelity length-ratio window (target_chars / source_chars). Inside → no
# penalty; outside → penalized up to saturation.
_FIDELITY_LOW = 0.7
_FIDELITY_HIGH = 1.8
_FIDELITY_SATURATION = 1.2


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


@dataclass
class CandidateScore:
    """A scored translation candidate."""

    text: str
    backend: str
    estimated_duration_s: float
    syllables: int
    slot_duration_s: float
    rate_factor: float
    duration_fit: float
    rate_penalty: float
    fidelity: float
    glossary_compliance: float
    score: float
    missing_glossary: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "backend": self.backend,
            "estimated_duration_s": round(self.estimated_duration_s, 3),
            "syllables": self.syllables,
            "rate_factor": round(self.rate_factor, 3),
            "duration_fit": round(self.duration_fit, 3),
            "rate_penalty": round(self.rate_penalty, 3),
            "fidelity": round(self.fidelity, 3),
            "glossary_compliance": round(self.glossary_compliance, 3),
            "score": round(self.score, 3),
            "missing_glossary": self.missing_glossary,
        }


def _glossary_compliance(text: str, terms: Sequence[str]) -> tuple[float, list[str]]:
    if not terms:
        return 1.0, []
    low = text.lower()
    missing = [t for t in terms if t.lower() not in low]
    compliance = 1.0 - (len(missing) / len(terms))
    return compliance, missing


def _fidelity(source_text: str, candidate: str) -> float:
    src = len(source_text.strip())
    tgt = len(candidate.strip())
    if src == 0:
        return 1.0 if tgt > 0 else 0.0
    ratio = tgt / src
    if _FIDELITY_LOW <= ratio <= _FIDELITY_HIGH:
        return 1.0
    if ratio < _FIDELITY_LOW:
        dev = _FIDELITY_LOW - ratio
    else:
        dev = ratio - _FIDELITY_HIGH
    return _clamp(1.0 - dev / _FIDELITY_SATURATION)


def _rate_penalty(rate_factor: float) -> float:
    if NATURAL_RATE_LOW <= rate_factor <= NATURAL_RATE_HIGH:
        return 0.0
    if rate_factor < NATURAL_RATE_LOW:
        dev = NATURAL_RATE_LOW - rate_factor
    else:
        dev = rate_factor - NATURAL_RATE_HIGH
    return _clamp(dev / _RATE_SATURATION)


def score_candidate(
    text: str,
    backend: str,
    slot_duration_s: float,
    source_text: str,
    lang: Optional[str],
    glossary_terms: Sequence[str] = (),
) -> CandidateScore:
    """Score one candidate translation against a slot."""
    est = estimate_duration(text, lang)
    slot = max(1e-3, float(slot_duration_s))
    rate_factor = est.seconds / slot if slot > 0 else 0.0

    duration_fit = _clamp(1.0 - abs(est.seconds - slot) / slot)
    rate_pen = _rate_penalty(rate_factor)
    fidelity = _fidelity(source_text, text)
    gloss, missing = _glossary_compliance(text, glossary_terms)

    score = (
        W_DURATION_FIT * duration_fit
        + W_RATE * (1.0 - rate_pen)
        + W_FIDELITY * fidelity
        + W_GLOSSARY * gloss
    )
    return CandidateScore(
        text=text,
        backend=backend,
        estimated_duration_s=est.seconds,
        syllables=est.syllables,
        slot_duration_s=slot,
        rate_factor=rate_factor,
        duration_fit=duration_fit,
        rate_penalty=rate_pen,
        fidelity=fidelity,
        glossary_compliance=gloss,
        score=round(score, 4),
        missing_glossary=missing,
    )


def select_candidate(
    candidates: Iterable[tuple[str, str]],
    slot_duration_s: float,
    source_text: str,
    lang: Optional[str],
    glossary_terms: Sequence[str] = (),
) -> tuple[Optional[CandidateScore], list[CandidateScore]]:
    """Score every ``(text, backend)`` candidate; return ``(best, all_scores)``.

    Ties on ``score`` are broken by higher ``duration_fit`` then ``fidelity``.
    Returns ``(None, [])`` when there are no candidates.
    """
    scored: list[CandidateScore] = []
    seen: set[str] = set()
    for text, backend in candidates:
        norm = (text or "").strip()
        if not norm or norm.lower() in seen:
            continue
        seen.add(norm.lower())
        scored.append(
            score_candidate(norm, backend, slot_duration_s, source_text, lang, glossary_terms)
        )
    if not scored:
        return None, []
    best = max(scored, key=lambda c: (c.score, c.duration_fit, c.fidelity))
    return best, scored
