"""Tests for timing-aware translation condensation (Failure #3).

Every segment shipped with n_candidates=1 (one literal translation), so the
timing selector had nothing shorter to choose and Portuguese — ~20-30%
longer than English — overflowed the slot (rate factors 1.24-1.66).

We generate condensed candidates by exploiting Portuguese pro-drop (dropping
redundant subject pronouns) and trimming discourse fillers, giving the
selector a natural, shorter option without changing meaning.
"""
from __future__ import annotations

from src.timing.condense import condense, condense_candidates
from src.timing.score import select_candidate


def test_drops_redundant_subject_pronoun():
    out = condense("Você está sinceramente tentando salvar o mundo?", "pt")
    assert out is not None
    assert out.lower().startswith("está")  # "Você" dropped, verb capitalised
    assert len(out) < len("Você está sinceramente tentando salvar o mundo?")


def test_trims_leading_filler():
    out = condense("Bem, estou tentando fazer coisas boas.", "pt")
    assert out is not None
    assert not out.lower().startswith("bem")
    assert "tentando fazer coisas boas" in out.lower()


def test_returns_none_when_nothing_to_cut():
    assert condense("Aqueça Marte.", "pt") is None


def test_non_pt_language_is_noop():
    assert condense("Are you trying to save the world?", "en") is None


def test_candidates_include_original_and_condensed():
    cands = condense_candidates("Você está tentando salvar o mundo?", "deep-translator:google", "pt")
    texts = [c[0] for c in cands]
    assert "Você está tentando salvar o mundo?" in texts
    assert any(t.lower().startswith("está") for t in texts)
    # The condensed variant is tagged so the report shows where it came from.
    assert any("condensed" in b for _, b in cands)


def test_selector_prefers_condensed_when_slot_is_tight():
    """A tight slot should select the shorter condensed candidate."""
    long_text = "Você está sinceramente tentando salvar o mundo inteiro?"
    cands = condense_candidates(long_text, "deep-translator:google", "pt")
    # 2.0s slot is too short for the literal line; the condensed one fits better.
    best, all_scores = select_candidate(cands, 2.0, source_text="Are you trying to save the world?", lang="pt")
    assert best is not None
    assert len(all_scores) >= 2
    assert best.estimated_duration_s <= max(c.estimated_duration_s for c in all_scores)
