"""Utility for entity detection and preservation across transcription and translation."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .logging import get_logger

LOG = get_logger("ai-dubbing.entities")


class EntityPreserver:
    """Detects and preserves named entities using a glossary and heuristics."""

    def __init__(self, glossary_path: Optional[Path] = None):
        self.glossary: Dict[str, Dict[str, Any]] = {}
        if glossary_path and glossary_path.exists():
            try:
                self.glossary = json.loads(glossary_path.read_text(encoding="utf-8"))
                LOG.info(f"Loaded glossary with {len(self.glossary)} entities from {glossary_path}")
            except Exception as exc:
                LOG.warning(f"Failed to load glossary: {exc}")
        
        self.placeholders: Dict[str, str] = {}
        self.reverse_placeholders: Dict[str, str] = {}

    def add_to_glossary(self, term: str, action: str = "preserve"):
        self.glossary[term] = {"action": action}

    def detect_entities(self, text: str) -> Set[str]:
        """Heuristic detection of named entities (capitalized words, etc.)."""
        # Look for sequences of capitalized words (2 or more)
        # Or single capitalized words that are not at the start of a sentence.
        entities = set()
        
        # Pattern for proper nouns (capitalized words not strictly at start of string)
        # This is a basic heuristic.
        words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        for w in words:
            if len(w) > 2:
                entities.add(w)
        
        # Add terms from glossary
        for term in self.glossary:
            if term.lower() in text.lower():
                # Find exact casing if possible
                match = re.search(re.escape(term), text, re.IGNORECASE)
                if match:
                    entities.add(match.group())
                else:
                    entities.add(term)
        
        return entities

    def protect(self, text: str) -> str:
        """Replace detected entities with placeholders to prevent translation."""
        self.placeholders = {}
        self.reverse_placeholders = {}
        
        entities = self.detect_entities(text)
        # Sort by length descending to avoid partial matches
        sorted_entities = sorted(list(entities), key=len, reverse=True)
        
        protected_text = text
        for i, ent in enumerate(sorted_entities):
            placeholder = f"__ENT_{i}__"
            self.placeholders[placeholder] = ent
            self.reverse_placeholders[ent] = placeholder
            # Use regex with word boundaries to replace
            protected_text = re.sub(r"\b" + re.escape(ent) + r"\b", placeholder, protected_text)
            
        return protected_text

    def restore(self, text: str) -> str:
        """Restore entities from placeholders after translation."""
        restored_text = text
        for placeholder, original in self.placeholders.items():
            restored_text = restored_text.replace(placeholder, original)
        return restored_text


def build_video_vocabulary(segments: List[Dict[str, Any]]) -> Set[str]:
    """Build a set of frequently occurring capitalized terms across the video."""
    all_text = " ".join(s.get("text", "") for s in segments)
    # Count occurrences of capitalized phrases
    candidates = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", all_text)
    counts = {}
    for c in candidates:
        counts[c] = counts.get(c, 0) + 1
    
    # Keep terms that appear multiple times or are in a known glossary
    # (Simplified for now: keep all with > 1 occurrence)
    return {c for c, count in counts.items() if count > 1 and len(c) > 3}
