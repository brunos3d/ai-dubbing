"""Tests for translation-candidate timing score & selection (spec §2.3)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.timing.score import score_candidate, select_candidate  # noqa: E402


def test_better_fitting_candidate_wins() -> None:
    """Given a 3 s slot, the candidate whose estimate is near 3 s should win
    over a much longer one that would force fast speech."""
    source = "Olá, tudo bem com você hoje?"
    # ~3 s of English at 4.4 syl/s ≈ 13 syllables.
    short = "Hi, how are you today?"           # ~6 syllables -> a bit short
    fitting = "Hello, how are you doing today my friend?"  # ~12 syllables
    overlong = (
        "Hello there, I would very much like to know how you are feeling "
        "on this particular day, my dear friend."
    )
    best, scores = select_candidate(
        [(short, "a"), (fitting, "b"), (overlong, "c")],
        slot_duration_s=3.0,
        source_text=source,
        lang="en",
    )
    assert best is not None
    assert best.text == fitting, [s.to_dict() for s in scores]
    # The overlong candidate must be penalized on rate (too fast to fit).
    overlong_score = next(s for s in scores if s.text == overlong)
    assert overlong_score.rate_penalty > 0.0


def test_glossary_noncompliance_is_penalized() -> None:
    a = score_candidate(
        "Spider-Man saves the city", "x", 2.0, "Homem-Aranha salva", "en",
        glossary_terms=["Spider-Man"],
    )
    b = score_candidate(
        "The hero saves the city", "x", 2.0, "Homem-Aranha salva", "en",
        glossary_terms=["Spider-Man"],
    )
    assert a.glossary_compliance == 1.0
    assert b.glossary_compliance < 1.0
    assert "Spider-Man" in b.missing_glossary


def test_weights_sum_to_one() -> None:
    from src.timing import score as s

    total = s.W_DURATION_FIT + s.W_RATE + s.W_FIDELITY + s.W_GLOSSARY
    assert abs(total - 1.0) < 1e-9


def test_empty_candidates_returns_none() -> None:
    best, scores = select_candidate([], 2.0, "x", "en")
    assert best is None and scores == []


def test_duplicate_candidates_deduplicated() -> None:
    best, scores = select_candidate(
        [("Hello world", "a"), ("hello world", "b"), ("Hello world", "c")],
        slot_duration_s=2.0,
        source_text="Olá mundo",
        lang="en",
    )
    assert len(scores) == 1
