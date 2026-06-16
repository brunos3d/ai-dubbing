"""Tests for the language-aware duration estimator (spec §2.1)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.timing.duration import (  # noqa: E402
    estimate_duration,
    estimate_syllables,
    speaking_rate,
)


def test_empty_text_is_zero() -> None:
    assert estimate_syllables("", "en") == 0
    est = estimate_duration("   ", "en")
    assert est.seconds == 0.0 and est.syllables == 0


def test_syllable_counts_english() -> None:
    # "hello world" -> hel-lo (2) + world (1) = 3
    assert estimate_syllables("hello world", "en") == 3
    # silent trailing e: "make" -> 1, not 2
    assert estimate_syllables("make", "en") == 1
    # "-le" ending keeps its syllable: "table" -> 2
    assert estimate_syllables("table", "en") == 2


def test_longer_text_takes_longer() -> None:
    short = estimate_duration("Hello.", "en")
    long = estimate_duration(
        "Hello there, this is a considerably longer sentence to speak.", "en"
    )
    assert long.seconds > short.seconds
    assert long.syllables > short.syllables


def test_language_rate_ordering() -> None:
    # Spanish is spoken faster (more syllables/sec) than German.
    assert speaking_rate("es") > speaking_rate("de")
    # Same syllable text estimated shorter in the faster language.
    text_es = "una frase de prueba bastante larga para medir"
    es = estimate_duration(text_es, "es")
    de = estimate_duration(text_es, "de")
    # With identical syllable counts, the faster rate yields a shorter time.
    assert es.syllables == de.syllables
    assert es.seconds < de.seconds


def test_pause_budget_adds_time() -> None:
    no_punct = estimate_duration("one two three four five", "en")
    with_punct = estimate_duration("one, two. three? four! five.", "en")
    assert with_punct.pause_s > 0.0
    assert with_punct.seconds > no_punct.seconds


def test_cjk_uses_char_proxy() -> None:
    # Each hanzi counts as one syllable.
    assert estimate_syllables("你好世界", "zh") == 4
    est = estimate_duration("你好世界", "zh")
    assert est.seconds > 0.0


def test_rate_override() -> None:
    slow = estimate_duration("hello world here we go", "en", rate_syl_s=2.0)
    fast = estimate_duration("hello world here we go", "en", rate_syl_s=8.0)
    assert slow.seconds > fast.seconds
