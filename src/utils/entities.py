"""Glossary-based entity preservation for translation.

The only source of entity-preservation truth is the user-supplied
glossary JSON.  No automatic entity detection runs.

Two glossary shapes are accepted:

* **Legacy / direct map**::

      {"Peter Parker": {"action": "preserve"}}

* **Entities list** (the format the dubbing CLI's ``glossary
  template`` command emits)::

      {"entities": [
          {"original": "Peter Parker", "replacement": "Peter Parker", "action": "preserve"},
          {"original": "Senior Partner", "replacement": "Sócio Sênior", "action": "replace"},
      ]}

  * ``action: "preserve"`` — the original term survives translation
    untouched (it is restored verbatim after the translator runs).
  * ``action: "replace"`` — the original term is protected during
    translation, then swapped for ``replacement`` afterwards.  This is
    the right primitive for terminology fixes that the translator
    cannot be trusted to make on its own (legal jargon, branded
    spellings, loanwords the target language should keep).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_glossary(raw: Any) -> Dict[str, Dict[str, Any]]:
    """Normalise the on-disk glossary shape into the internal
    ``{term: {action, replacement?}}`` map.

    The internal representation always uses the term as the key so the
    placeholder loop in :meth:`EntityPreserver.protect` can stay
    term-centric (longest-first iteration, single regex compile).
    """
    if not isinstance(raw, dict):
        return {}

    entities = raw.get("entities")
    if isinstance(entities, list):
        out: Dict[str, Dict[str, Any]] = {}
        for entry in entities:
            if not isinstance(entry, dict):
                continue
            term = entry.get("original")
            if not isinstance(term, str) or not term:
                continue
            action = entry.get("action", "preserve")
            rec: Dict[str, Any] = {"action": action}
            replacement = entry.get("replacement")
            if action == "replace" and isinstance(replacement, str) and replacement:
                rec["replacement"] = replacement
            out[term] = rec
        return out

    # Legacy direct-map shape: {"Term": {"action": "preserve"|"replace", "replacement": "..."}}
    out = {}
    for term, meta in raw.items():
        if not isinstance(term, str) or not term:
            continue
        if not isinstance(meta, dict):
            continue
        action = meta.get("action", "preserve")
        rec = {"action": action}
        if action == "replace":
            replacement = meta.get("replacement")
            if isinstance(replacement, str) and replacement:
                rec["replacement"] = replacement
        out[term] = rec
    return out


class EntityPreserver:
    """Substitutes glossary terms with placeholders before translation,
    then restores them after.

    A glossary entry maps ``"Exact Term"`` to ``{"action": "preserve"}``
    or ``{"action": "replace", "replacement": "..."}``.  Only the
    literal terms in the glossary are protected; no regex heuristics,
    no sentence-capitalization guesses, no video-wide vocabulary
    extraction.

    When the glossary is empty (the common case), :meth:`protect` is a
    no-op pass-through, and :meth:`restore` is a no-op pass-through. The
    translator sees the original text verbatim.
    """

    def __init__(self, glossary_path: Optional[Path] = None):
        self.glossary: Dict[str, Dict[str, Any]] = {}
        if glossary_path and glossary_path.exists():
            try:
                raw = json.loads(glossary_path.read_text(encoding="utf-8"))
                self.glossary = _load_glossary(raw)
            except Exception as exc:
                # Bad glossary must not crash the pipeline.
                self.glossary = {}
                from .logging import get_logger
                get_logger("ai-dubbing.entities").warning(
                    f"Failed to load glossary {glossary_path}: {exc}"
                )

        self._loaded_count = len(self.glossary)
        # ``placeholders`` maps the in-text placeholder token to the
        # string that :meth:`restore` should swap it back to.  For
        # ``preserve`` actions this is the original term; for
        # ``replace`` actions this is the configured replacement.
        # Downstream consumers (e.g. ``translate.py`` recording
        # ``entities_preserved``) see the resolved post-translation
        # value, which is the more useful thing to log.
        self.placeholders: Dict[str, str] = {}

    @property
    def is_active(self) -> bool:
        """True when the glossary has at least one entry."""
        return self._loaded_count > 0

    def add_to_glossary(self, term: str, action: str = "preserve") -> None:
        self.glossary[term] = {"action": action}
        self._loaded_count = len(self.glossary)

    def _restore_target(self, term: str) -> str:
        """Return the string the placeholder should be swapped back to.

        For ``preserve``, that is the original term.  For ``replace``,
        it is the configured replacement (the translated/locale-specific
        term the user wants to enforce).
        """
        meta = self.glossary.get(term, {})
        if meta.get("action") == "replace":
            repl = meta.get("replacement")
            if isinstance(repl, str) and repl:
                return repl
        return term

    def protect(self, text: str) -> str:
        """Replace glossary terms with placeholders.

        If the glossary is empty, returns ``text`` unchanged.
        """
        self.placeholders = {}
        if not self.is_active:
            return text

        protected = text
        # Sort by length descending so multi-word terms win over their
        # sub-phrases (e.g. "Peter Parker" before "Peter").
        for i, term in enumerate(sorted(self.glossary.keys(), key=len, reverse=True)):
            placeholder = f"__ENT_{i}__"
            self.placeholders[placeholder] = self._restore_target(term)
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            protected = pattern.sub(placeholder, protected)
        return protected

    def restore(self, text: str) -> str:
        """Replace placeholders with their target term (original for
        ``preserve``, ``replacement`` for ``replace``).

        If the glossary was empty, returns ``text`` unchanged.
        """
        if not self.placeholders:
            return text
        for placeholder, target in self.placeholders.items():
            text = text.replace(placeholder, target)
        return text
