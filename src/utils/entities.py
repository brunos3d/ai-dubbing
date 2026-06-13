"""Glossary-based entity preservation for translation.

The only source of entity-preservation truth is the user-supplied
glossary JSON.  No automatic entity detection runs.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional


class EntityPreserver:
    """Substitutes glossary terms with placeholders before translation,
    then restores them after.

    A glossary entry maps ``"Exact Term"`` to ``{"action": "preserve"}``.
    Only the literal terms in the glossary are protected; no regex
    heuristics, no sentence-capitalization guesses, no video-wide
    vocabulary extraction.

    When the glossary is empty (the common case), :meth:`protect` is a
    no-op pass-through, and :meth:`restore` is a no-op pass-through. The
    translator sees the original text verbatim.
    """

    def __init__(self, glossary_path: Optional[Path] = None):
        self.glossary: Dict[str, Dict[str, Any]] = {}
        if glossary_path and glossary_path.exists():
            try:
                self.glossary = json.loads(glossary_path.read_text(encoding="utf-8"))
            except Exception as exc:
                # Bad glossary must not crash the pipeline.
                self.glossary = {}
                from .logging import get_logger
                get_logger("ai-dubbing.entities").warning(
                    f"Failed to load glossary {glossary_path}: {exc}"
                )

        self._loaded_count = len(self.glossary)
        self.placeholders: Dict[str, str] = {}

    @property
    def is_active(self) -> bool:
        """True when the glossary has at least one entry."""
        return self._loaded_count > 0

    def add_to_glossary(self, term: str, action: str = "preserve") -> None:
        self.glossary[term] = {"action": action}
        self._loaded_count = len(self.glossary)

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
            self.placeholders[placeholder] = term
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            protected = pattern.sub(placeholder, protected)
        return protected

    def restore(self, text: str) -> str:
        """Replace placeholders with the original glossary terms.

        If the glossary was empty, returns ``text`` unchanged.
        """
        if not self.placeholders:
            return text
        for placeholder, original in self.placeholders.items():
            text = text.replace(placeholder, original)
        return text
