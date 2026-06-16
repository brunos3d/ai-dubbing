"""Timeline v1 — the canonical per-segment representation (spec §3).

The Timeline is a structured, queryable document that gathers everything the
pipeline knows about each dubbing segment into one place: who spoke, when,
the source and target text, the glossary state, the timing plan, the prosody
descriptors, the generation/render state, and review metadata.

Design constraints for v1 (deliberately conservative):

* **Derived, not authoritative.** The Timeline is *rebuilt from* the existing
  artifacts (diarization, transcript, translation, generate/align manifests)
  rather than replacing them. Stages keep reading the legacy JSON files; a
  later cycle can flip the dependency so stages read the Timeline directly.
* **Coexists & never breaks compatibility.** It is persisted at
  ``<workspace>/timeline.json`` and registered as a *derived* artifact, so it
  never participates in invalidation. An absent ``timeline.json`` (old
  workspace) is fine — it is regenerated on the next run.
* **The home of the segment render key** (Phase 4): :func:`segment_render_key`
  is the single definition of "what makes a segment's synthesized audio
  unique", reused by the generate stage for segment-level reuse.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1

# A segment's timing score below this is surfaced for human review.
_LOW_SCORE_THRESHOLD = 0.6


def segment_render_key(
    *,
    target_text: str,
    speaker: str,
    target_language: str,
    voice_profile_hash: str = "",
    tts_model_id: str = "",
    slot_duration_s: float = 0.0,
    prosody_signature: str = "",
) -> str:
    """Stable 16-hex hash of everything that affects a segment's TTS render.

    Changing the translated text, the speaker/voice, the target language, the
    model, the slot duration, or the prosody conditioning yields a new key —
    which is exactly what the generate stage uses to decide whether a segment
    can be reused or must be re-synthesized (Phase 4).
    """
    payload = "||".join(
        [
            (target_text or "").strip(),
            speaker or "",
            target_language or "",
            voice_profile_hash or "",
            tts_model_id or "",
            f"{round(float(slot_duration_s or 0.0), 3)}",
            prosody_signature or "",
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class TimelineSegment:
    """One dubbing segment — the atomic unit of translation/timing/synthesis."""

    id: str
    speaker: str
    start: float
    end: float
    source_text: str = ""
    target_text: str = ""
    glossary_hits: List[str] = field(default_factory=list)
    timing: Dict[str, Any] = field(default_factory=dict)
    prosody: Dict[str, Any] = field(default_factory=dict)
    generation: Dict[str, Any] = field(default_factory=dict)
    review: Dict[str, Any] = field(default_factory=dict)
    render_key: str = ""
    is_non_speech: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TimelineSegment":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Timeline:
    """The canonical per-segment document for a workspace."""

    schema_version: int
    workspace_id: str
    language_pair: str
    segments: List[TimelineSegment] = field(default_factory=list)

    # -- serialization ---------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "workspace_id": self.workspace_id,
            "language_pair": self.language_pair,
            "segments": [s.to_dict() for s in self.segments],
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> Optional["Timeline"]:
        path = Path(path)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return cls(
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
            workspace_id=str(data.get("workspace_id", "")),
            language_pair=str(data.get("language_pair", "")),
            segments=[TimelineSegment.from_dict(s) for s in data.get("segments", [])],
        )

    def summary(self) -> Dict[str, Any]:
        """A compact rollup for ``workspace inspect`` and logs."""
        n = len(self.segments)
        generated = sum(1 for s in self.segments if s.generation.get("render_path"))
        reused = sum(1 for s in self.segments if s.generation.get("reused"))
        flagged = sum(1 for s in self.segments if s.review.get("low_confidence"))
        scores = [
            s.timing.get("score")
            for s in self.segments
            if isinstance(s.timing.get("score"), (int, float))
        ]
        mean_score = round(sum(scores) / len(scores), 3) if scores else None
        return {
            "segments": n,
            "generated": generated,
            "reused": reused,
            "flagged_for_review": flagged,
            "mean_timing_score": mean_score,
        }


# ---------------------------------------------------------------------------
# Build-from-workspace
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def build_from_workspace(root: Path) -> Timeline:
    """Rebuild a :class:`Timeline` from the artifacts present in ``root``.

    Tolerant of missing pieces: a freshly-prepared workspace (no generate yet)
    yields a Timeline with text + timing but empty generation state; an old
    workspace with only a transcript still yields useful segments.
    """
    root = Path(root)
    meta = _read_json(root / "metadata.json") or {}
    source = meta.get("source", {})
    workspace_id = meta.get("workspace_id", root.name)
    src_lang = source.get("source_language", "")
    tgt_lang = source.get("target_language", "")

    # Prefer the translated transcript (carries source_text + target text +
    # timing); fall back to the raw transcript for a prepared-but-untranslated
    # workspace.
    translated = _read_json(root / "translation" / "translated_transcript.json")
    transcript = _read_json(root / "transcription" / "transcript.json")
    base_segments = translated if isinstance(translated, list) else (
        transcript if isinstance(transcript, list) else []
    )
    have_translation = isinstance(translated, list)

    # Generation / alignment state, indexed by segment order.
    aligned = _read_json(root / "aligned_manifest.json")
    gen_manifest = _read_json(root / "generated_segments" / "manifest.json")
    render_by_index: Dict[int, Dict[str, Any]] = {}
    for manifest in (gen_manifest, aligned):
        if isinstance(manifest, list):
            for entry in manifest:
                if isinstance(entry, dict) and "index" in entry:
                    render_by_index.setdefault(int(entry["index"]), {}).update(entry)

    # Prosody descriptors (Phase 5), if present.
    prosody_doc = _read_json(root / "prosody" / "prosody.json")
    prosody_by_index: Dict[int, Dict[str, Any]] = {}
    if isinstance(prosody_doc, dict) and isinstance(prosody_doc.get("segments"), list):
        for entry in prosody_doc["segments"]:
            if isinstance(entry, dict) and "index" in entry:
                prosody_by_index[int(entry["index"])] = entry

    segments: List[TimelineSegment] = []
    for i, seg in enumerate(base_segments):
        if not isinstance(seg, dict):
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        speaker = seg.get("speaker", "speaker_unknown")
        source_text = seg.get("source_text", seg.get("text", "") if not have_translation else "")
        target_text = seg.get("text", "") if have_translation else ""
        timing = dict(seg.get("timing", {})) if isinstance(seg.get("timing"), dict) else {}
        glossary_hits = list(seg.get("entities_preserved", []) or [])

        render = render_by_index.get(i, {})
        generation: Dict[str, Any] = {}
        if render:
            generation = {
                "render_path": render.get("aligned_path") or render.get("path"),
                "generated_duration": render.get("generated_duration"),
                "aligned_duration": render.get("aligned_duration"),
                "alignment_method": render.get("alignment_method"),
                "reused": bool(render.get("reused", False)),
            }
            if render.get("render_key"):
                generation["render_key"] = render["render_key"]

        prosody = prosody_by_index.get(i, {})

        score = timing.get("score")
        low_conf = isinstance(score, (int, float)) and score < _LOW_SCORE_THRESHOLD
        review = {
            "low_confidence": bool(low_conf),
            "locked": bool(seg.get("locked", False)),
        }

        voice_hash = ""
        prof = root / "speaker_profiles" / speaker / "primary" / "metadata.json"
        prof_meta = _read_json(prof)
        if isinstance(prof_meta, dict):
            voice_hash = prof_meta.get("voice_profile_hash", "")

        render_key = segment_render_key(
            target_text=target_text or source_text,
            speaker=speaker,
            target_language=tgt_lang,
            voice_profile_hash=voice_hash,
            slot_duration_s=max(0.0, end - start),
            prosody_signature=prosody.get("signature", ""),
        )

        segments.append(
            TimelineSegment(
                id=f"seg_{i + 1:04d}",
                speaker=speaker,
                start=round(start, 3),
                end=round(end, 3),
                source_text=source_text,
                target_text=target_text,
                glossary_hits=glossary_hits,
                timing=timing,
                prosody=prosody,
                generation=generation,
                review=review,
                render_key=render_key,
                is_non_speech=bool(seg.get("is_non_speech", False)),
            )
        )

    return Timeline(
        schema_version=SCHEMA_VERSION,
        workspace_id=workspace_id,
        language_pair=f"{src_lang}->{tgt_lang}",
        segments=segments,
    )


def refresh_timeline(root: Path) -> Timeline:
    """Rebuild and persist ``<root>/timeline.json``; return the Timeline."""
    tl = build_from_workspace(root)
    tl.save(Path(root) / "timeline.json")
    return tl
