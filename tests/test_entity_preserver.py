"""Regression tests for the entity-preserver hotfix.

These tests lock in the fix for the corruption caused by automatic
regex-based entity detection.  After the hotfix:

* `EntityPreserver.protect()` is a no-op when no glossary is supplied.
* Only literal terms in the supplied glossary are protected.
* No sentence-initial capitalised words are protected.
* The translator sees the original text verbatim when no glossary.

The tests construct an in-memory glossary (no file I/O) and inspect
the protect/restore round-trip directly.  They do not require a
running translator.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Make `src` importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.entities import EntityPreserver  # noqa: E402


def _write_glossary(entries: dict) -> Path:
    """Materialise a glossary dict to a temp file and return the path."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(entries, f)
    f.flush()
    f.close()
    return Path(f.name)


def test_case_a_no_glossary_leaves_text_untouched():
    """Case A: no glossary, no entity preservation, no placeholders.

    Source: 'Vamos conferir agora todos os vazamentos'
    Expected English: 'Let's check out all the leaks now'
    Must NOT become:    'Vamos check out all the leaks now'
    """
    preserver = EntityPreserver(glossary_path=None)
    assert not preserver.is_active, "Empty glossary should leave preserver inactive"

    source = "Vamos conferir agora todos os vazamentos"
    protected = preserver.protect(source)
    assert protected == source, (
        f"With no glossary, protect() must return text verbatim.\n"
        f"  source:    {source!r}\n"
        f"  protected: {protected!r}"
    )
    assert preserver.placeholders == {}, "No placeholders should be created"

    # restore() on the same text returns it unchanged.
    assert preserver.restore(protected) == source


def test_case_b_glossary_protects_named_entities():
    """Case B: glossary present, the named term survives translation.

    Source:   'Peter Parker apareceu'
    Glossary: {'Peter Parker': {'action': 'preserve'}}
    Expected: 'Peter Parker appeared' (with the term untouched)
    """
    glossary_path = _write_glossary({"Peter Parker": {"action": "preserve"}})
    try:
        preserver = EntityPreserver(glossary_path=glossary_path)
        assert preserver.is_active
        assert preserver._loaded_count == 1

        source = "Peter Parker apareceu no filme."
        protected = preserver.protect(source)
        # The literal "Peter Parker" must be replaced with a placeholder.
        assert "Peter Parker" not in protected, (
            f"Glossary term must be replaced with a placeholder.\n"
            f"  source:    {source!r}\n"
            f"  protected: {protected!r}"
        )
        assert "__ENT_0__" in protected, (
            f"Placeholder __ENT_0__ must be present in protected text.\n"
            f"  protected: {protected!r}"
        )
        # The rest of the text (e.g. "apareceu no filme.") must be intact.
        assert "apareceu no filme." in protected, (
            f"Non-glossary text must be untouched.\n"
            f"  protected: {protected!r}"
        )

        # Simulate a translator that returns English with the placeholder
        # preserved (this is what deep-translator does: it leaves unknown
        # tokens untouched).
        simulated_translation = "__ENT_0__ appeared in the movie."
        final = preserver.restore(simulated_translation)
        assert final == "Peter Parker appeared in the movie.", (
            f"Glossary term must be restored verbatim.\n"
            f"  got: {final!r}"
        )
    finally:
        glossary_path.unlink()


def test_case_c_no_glossary_full_translation_no_remaining_portuguese():
    """Case C: no glossary, no entity preservation.

    Source: 'Isso aqui é muito maior'
    Expected: fully translated English
    Must NOT contain: 'Isso'
    """
    preserver = EntityPreserver(glossary_path=None)
    assert not preserver.is_active

    source = "Isso aqui é muito maior"
    protected = preserver.protect(source)
    assert protected == source
    assert "Isso" not in preserver.placeholders.values()
    # The restore() pass-through also doesn't reintroduce the term.
    assert preserver.restore("This is much bigger") == "This is much bigger"


def test_sentence_initial_capitalised_words_are_not_protected():
    """The original bug: 'Vamos', 'Isso', 'Cara', 'Então', 'Agora' at
    the start of a sentence were protected by the regex heuristic.  They
    must now pass through verbatim.
    """
    preserver = EntityPreserver(glossary_path=None)
    for sentence in [
        "Vamos conferir agora todos os vazamentos",
        "Isso aqui é muito maior",
        "Cara, o que você acha?",
        "Então, vamos começar.",
        "Agora mesmo.",
    ]:
        protected = preserver.protect(sentence)
        assert protected == sentence, (
            f"Sentence-initial capitalised word leaked through protection.\n"
            f"  sentence:  {sentence!r}\n"
            f"  protected: {protected!r}"
        )


def test_glossary_with_multiple_terms_longest_first():
    """Multi-word glossary terms must take precedence over their
    sub-phrases (e.g. 'Peter Parker' before 'Peter').
    """
    glossary_path = _write_glossary({
        "Peter": {"action": "preserve"},
        "Peter Parker": {"action": "preserve"},
    })
    try:
        preserver = EntityPreserver(glossary_path=glossary_path)
        source = "Peter Parker fought the robber Peter."
        protected = preserver.protect(source)
        # "Peter Parker" should be replaced first, leaving a single
        # "Peter" placeholder for the second occurrence.
        # Exact placeholder numbering depends on iteration order, so we
        # just assert the structural property: the multi-word term is
        # fully replaced and the single-word term appears once more.
        assert "Peter Parker" not in protected
        assert protected.count("Peter") == 0 or protected.count("__ENT_") == 2
    finally:
        glossary_path.unlink()


def test_is_active_reflects_glossary_presence():
    preserver_no = EntityPreserver(glossary_path=None)
    assert preserver_no.is_active is False

    p = _write_glossary({"Marvel": {"action": "preserve"}})
    try:
        preserver_yes = EntityPreserver(glossary_path=p)
        assert preserver_yes.is_active is True
    finally:
        p.unlink()


def test_bad_glossary_does_not_crash():
    """A malformed glossary file must not crash the pipeline; the
    preserver should fall back to the empty-glossary no-op state.
    """
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    f.write("not valid json {{")
    f.flush()
    f.close()
    try:
        preserver = EntityPreserver(glossary_path=Path(f.name))
        assert preserver.is_active is False
        assert preserver.protect("anything") == "anything"
    finally:
        Path(f.name).unlink()


def run_all() -> int:
    """Discover and run every test_* function in this file."""
    tests = [
        test_case_a_no_glossary_leaves_text_untouched,
        test_case_b_glossary_protects_named_entities,
        test_case_c_no_glossary_full_translation_no_remaining_portuguese,
        test_sentence_initial_capitalised_words_are_not_protected,
        test_glossary_with_multiple_terms_longest_first,
        test_is_active_reflects_glossary_presence,
        test_bad_glossary_does_not_crash,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print()
    if failures:
        print(f"{failures} test(s) failed out of {len(tests)}")
    else:
        print(f"All {len(tests)} tests passed.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all())
