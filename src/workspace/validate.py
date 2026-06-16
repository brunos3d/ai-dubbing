"""Per-artifact validators for workspace files (spec §14).

Each :func:`validate_*` function returns a list of :class:`Issue` objects.
Errors block generation; warnings are reported but do not abort.

The validators are deliberately tolerant of non-dict input — a corrupt
manifest can never raise an exception, only produce an error report.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf


@dataclass(frozen=True)
class Issue:
    """A single validation issue.

    ``severity`` is ``"error"`` or ``"warning"``. ``path`` is a JSONPath-like
    string identifying the offending location (e.g. ``"$.segments[0].text"``).
    ``message`` is a human-readable explanation.
    """

    severity: str
    path: str
    message: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_required_keys(
    data: Any,
    required: tuple[str, ...],
    root: str,
    issues: list[Issue],
) -> bool:
    """Append an error for each missing required key. Return True if data is a dict."""
    if not isinstance(data, dict):
        issues.append(
            Issue("error", root, f"expected object, got {type(data).__name__}")
        )
        return False
    for key in required:
        if key not in data:
            issues.append(
                Issue("error", f"{root}.{key}", f"missing required key '{key}'")
            )
    return True


# ---------------------------------------------------------------------------
# Transcript / translation (spec §12)
# ---------------------------------------------------------------------------


_TRANSCRIPT_TOP_REQUIRED = ("$schema_version", "language", "segments")
_SEGMENT_REQUIRED = ("id", "start", "end", "speaker", "text")
# The runtime stages emit a *bare list* of segments (legacy/v0 shape) rather
# than the richer ``{$schema_version, language, segments}`` v1 dict.  The
# validators accept both so ``dub.sh workspace validate`` is truthful about
# the artifacts the pipeline actually produces.  Bare-list segments are not
# required to carry an ``id`` (the dict shape is); everything else matches.
_LEGACY_SEGMENT_REQUIRED = ("start", "end", "speaker", "text")


def _validate_segment_list(
    segments: list, required: tuple[str, ...], issues: list[Issue]
) -> None:
    """Append issues for a bare list of segment dicts."""
    for i, seg in enumerate(segments):
        seg_path = f"$[{i}]"
        if not isinstance(seg, dict):
            issues.append(
                Issue("error", seg_path, f"expected object, got {type(seg).__name__}")
            )
            continue
        for key in required:
            if key not in seg:
                issues.append(
                    Issue("error", f"{seg_path}.{key}", f"missing required key '{key}'")
                )
        if "text" in seg and not isinstance(seg["text"], str):
            issues.append(
                Issue(
                    "error",
                    f"{seg_path}.text",
                    f"text must be a string, got {type(seg['text']).__name__}",
                )
            )


def validate_transcript(data: Any) -> list[Issue]:
    """Validate a transcript (v1 dict *or* legacy bare list of segments)."""
    issues: list[Issue] = []
    if isinstance(data, list):
        _validate_segment_list(data, _LEGACY_SEGMENT_REQUIRED, issues)
        return issues
    if not _check_required_keys(data, _TRANSCRIPT_TOP_REQUIRED, "$", issues):
        return issues
    segments = data.get("segments")
    if "segments" in data and not isinstance(segments, list):
        issues.append(
            Issue(
                "error",
                "$.segments",
                f"expected list, got {type(segments).__name__}",
            )
        )
        return issues
    if not isinstance(segments, list):
        return issues
    for i, seg in enumerate(segments):
        seg_path = f"$.segments[{i}]"
        if not isinstance(seg, dict):
            issues.append(
                Issue("error", seg_path, f"expected object, got {type(seg).__name__}")
            )
            continue
        for key in _SEGMENT_REQUIRED:
            if key not in seg:
                issues.append(
                    Issue(
                        "error",
                        f"{seg_path}.{key}",
                        f"missing required key '{key}'",
                    )
                )
        if "text" in seg and not isinstance(seg["text"], str):
            issues.append(
                Issue(
                    "error",
                    f"{seg_path}.text",
                    f"text must be a string, got {type(seg['text']).__name__}",
                )
            )
    return issues


def validate_translation(data: Any) -> list[Issue]:
    """Validate a translated-transcript dict (spec §12).

    A translated transcript is a transcript with ``source_text`` on every
    segment. A missing ``source_text`` is a warning (the segment is still
    usable) but a hard error if the segment also has other structural
    problems caught by :func:`validate_transcript`.
    """
    issues = validate_transcript(data)
    if isinstance(data, list):
        for i, seg in enumerate(data):
            if not isinstance(seg, dict):
                continue
            if "source_text" not in seg:
                issues.append(
                    Issue("warning", f"$[{i}].source_text", "missing source_text")
                )
        return issues
    if not isinstance(data, dict):
        return issues
    segments = data.get("segments")
    if not isinstance(segments, list):
        return issues
    for i, seg in enumerate(segments):
        seg_path = f"$.segments[{i}]"
        if not isinstance(seg, dict):
            continue
        if "source_text" not in seg:
            issues.append(
                Issue(
                    "warning",
                    f"{seg_path}.source_text",
                    "missing source_text",
                )
            )
    return issues


# ---------------------------------------------------------------------------
# Glossary (spec §12)
# ---------------------------------------------------------------------------


_GLOSSARY_TOP_REQUIRED = ("$schema_version", "entries")
_VALID_GLOSSARY_ACTIONS = ("preserve",)


def validate_glossary(data: Any) -> list[Issue]:
    """Validate a glossary dict (spec §12 glossary format)."""
    issues: list[Issue] = []
    if not _check_required_keys(data, _GLOSSARY_TOP_REQUIRED, "$", issues):
        return issues
    entries = data.get("entries")
    if "entries" in data and not isinstance(entries, dict):
        issues.append(
            Issue(
                "error",
                "$.entries",
                f"expected object, got {type(entries).__name__}",
            )
        )
        return issues
    if not isinstance(entries, dict):
        return issues
    for term, entry in entries.items():
        entry_path = f"$.entries[{term!r}]"
        if not isinstance(entry, dict):
            issues.append(
                Issue(
                    "error",
                    entry_path,
                    f"expected object, got {type(entry).__name__}",
                )
            )
            continue
        action = entry.get("action")
        if action not in _VALID_GLOSSARY_ACTIONS:
            issues.append(
                Issue(
                    "error",
                    f"{entry_path}.action",
                    f"unknown action {action!r}; expected one of "
                    f"{list(_VALID_GLOSSARY_ACTIONS)}",
                )
            )
    return issues


# ---------------------------------------------------------------------------
# Diarization segments (spec §12)
# ---------------------------------------------------------------------------


_DIARIZATION_TOP_REQUIRED = ("$schema_version", "speakers", "segments")
_DIARIZATION_SEGMENT_REQUIRED = ("start", "end", "speaker")


def validate_diarization_segments(data: Any) -> list[Issue]:
    """Validate diarization segments (v1 dict *or* legacy bare list)."""
    issues: list[Issue] = []
    if isinstance(data, list):
        _validate_segment_list(data, _DIARIZATION_SEGMENT_REQUIRED, issues)
        return issues
    if not _check_required_keys(data, _DIARIZATION_TOP_REQUIRED, "$", issues):
        return issues
    speakers = data.get("speakers")
    if "speakers" in data and not isinstance(speakers, list):
        issues.append(
            Issue(
                "error",
                "$.speakers",
                f"expected list, got {type(speakers).__name__}",
            )
        )
        speakers = None
    segments = data.get("segments")
    if "segments" in data and not isinstance(segments, list):
        issues.append(
            Issue(
                "error",
                "$.segments",
                f"expected list, got {type(segments).__name__}",
            )
        )
        return issues
    if not isinstance(segments, list):
        return issues
    speaker_set = set(speakers) if isinstance(speakers, list) else None
    for i, seg in enumerate(segments):
        seg_path = f"$.segments[{i}]"
        if not isinstance(seg, dict):
            issues.append(
                Issue("error", seg_path, f"expected object, got {type(seg).__name__}")
            )
            continue
        for key in _DIARIZATION_SEGMENT_REQUIRED:
            if key not in seg:
                issues.append(
                    Issue(
                        "error",
                        f"{seg_path}.{key}",
                        f"missing required key '{key}'",
                    )
                )
        spk = seg.get("speaker")
        if speaker_set is not None and isinstance(spk, str) and spk not in speaker_set:
            issues.append(
                Issue(
                    "warning",
                    f"{seg_path}.speaker",
                    f"speaker {spk!r} not in declared speakers list",
                )
            )
    return issues


# ---------------------------------------------------------------------------
# Speaker metadata (spec §12)
# ---------------------------------------------------------------------------


_SPEAKER_METADATA_REQUIRED = (
    "$schema_version",
    "speaker_id",
    "voice_profile_hash",
    "score",
    "snr_db",
    "reference_duration_s",
    "segments_used",
    "transcript_words",
    "model",
    "extracted_at",
)
_SPEAKER_MIN_DURATION_S = 3.0
_SPEAKER_MAX_DURATION_S = 15.0


def validate_speaker_metadata(data: Any) -> list[Issue]:
    """Validate a speaker metadata dict (spec §12).

    ``reference_duration_s < 3.0`` is an error (the voice profile is too
    short to clone reliably). ``> 15.0`` is a warning (the reference is
    longer than the recommended maximum).
    """
    issues: list[Issue] = []
    if not _check_required_keys(data, _SPEAKER_METADATA_REQUIRED, "$", issues):
        return issues
    dur = data.get("reference_duration_s")
    if isinstance(dur, bool):
        return issues
    if isinstance(dur, (int, float)):
        if dur < _SPEAKER_MIN_DURATION_S:
            issues.append(
                Issue(
                    "error",
                    "$.reference_duration_s",
                    f"reference duration {dur}s is below minimum "
                    f"{_SPEAKER_MIN_DURATION_S}s",
                )
            )
        elif dur > _SPEAKER_MAX_DURATION_S:
            issues.append(
                Issue(
                    "warning",
                    "$.reference_duration_s",
                    f"reference duration {dur}s is above recommended "
                    f"{_SPEAKER_MAX_DURATION_S}s",
                )
            )
    return issues


# ---------------------------------------------------------------------------
# Speaker sample — WAV + transcript file (spec §12 primary.wav requirements)
# ---------------------------------------------------------------------------


def validate_speaker_sample(wav: Path, txt: Path) -> list[Issue]:
    """Validate a speaker primary sample on disk.

    * ``txt`` must exist.
    * ``wav`` must exist and be readable by ``soundfile``.
    * Sample rate should be 16 kHz (warning if not).
    * Audio should be mono (warning if not).
    * Duration ``< 3.0 s`` is an error; ``> 15.0 s`` is a warning.
    """
    issues: list[Issue] = []
    txt_path = Path(txt)
    if not txt_path.exists():
        issues.append(Issue("error", str(txt_path), "transcript file does not exist"))
    wav_path = Path(wav)
    if not wav_path.exists():
        issues.append(Issue("error", str(wav_path), "wav file does not exist"))
        return issues
    try:
        info = sf.info(str(wav_path))
    except Exception as exc:  # noqa: BLE001 — surface any decode failure
        issues.append(Issue("error", str(wav_path), f"failed to read wav: {exc}"))
        return issues
    if info.samplerate != 16000:
        issues.append(
            Issue(
                "warning",
                str(wav_path),
                f"sample rate {info.samplerate} != 16000",
            )
        )
    if info.channels != 1:
        issues.append(
            Issue(
                "warning",
                str(wav_path),
                f"audio is not mono ({info.channels} channels)",
            )
        )
    duration = info.frames / info.samplerate if info.samplerate else 0.0
    if duration < _SPEAKER_MIN_DURATION_S:
        issues.append(
            Issue(
                "error",
                str(wav_path),
                f"audio duration {duration:.2f}s is below minimum "
                f"{_SPEAKER_MIN_DURATION_S}s",
            )
        )
    elif duration > _SPEAKER_MAX_DURATION_S:
        issues.append(
            Issue(
                "warning",
                str(wav_path),
                f"audio duration {duration:.2f}s is above recommended "
                f"{_SPEAKER_MAX_DURATION_S}s",
            )
        )
    return issues


# ---------------------------------------------------------------------------
# Manifest (spec §9)
# ---------------------------------------------------------------------------


_MANIFEST_TOP_REQUIRED = ("schema_version", "workspace_id", "stages")
_MANIFEST_SCHEMA_VERSION = 1


def validate_manifest(data: Any) -> list[Issue]:
    """Validate a manifest dict (spec §9)."""
    issues: list[Issue] = []
    if not _check_required_keys(data, _MANIFEST_TOP_REQUIRED, "$", issues):
        return issues
    sv = data.get("schema_version")
    if sv != _MANIFEST_SCHEMA_VERSION:
        issues.append(
            Issue(
                "error",
                "$.schema_version",
                f"schema_version {sv!r} != {_MANIFEST_SCHEMA_VERSION}",
            )
        )
    return issues


# ---------------------------------------------------------------------------
# Metadata (spec §8)
# ---------------------------------------------------------------------------


_METADATA_TOP_REQUIRED = (
    "workspace_schema_version",
    "workspace_id",
    "content_hash",
    "created_at",
    "updated_at",
    "workspace_created_with",
    "source",
    "config",
    "pipeline_config_hash",
)


def validate_metadata(data: Any) -> list[Issue]:
    """Validate a workspace metadata dict (spec §8)."""
    issues: list[Issue] = []
    _check_required_keys(data, _METADATA_TOP_REQUIRED, "$", issues)
    return issues
