"""Workspace invalidation DAG (spec §10, release blocker).

Given a :class:`~src.workspace.manifest.Manifest` and the current contents of
the workspace on disk, decide which stages need to re-run. The algorithm is
defined verbatim in spec §10; this module is the executable form of that
spec.

Conceptual model:

* Each stage declares inputs and outputs. An output is a file the stage
  produced; an input is a file it read.
* "Editable" outputs are user-facing artefacts (translated transcript,
  glossary, primary speaker sample, transcript) — if the user edits one, we
  trust the edit and only mark the *consumer* of that file as stale (we do
  not walk upstream to the original producer).
* "Non-editable" outputs (everything else) — if the recorded hash no longer
  matches the file on disk, we mark the *producer* stale so it can re-create
  the file.
* "Derived" outputs (embeddings, large machine-generated blobs) are
  intentionally excluded from invalidation: they are regenerated as a
  side-effect of the stage that owns them and are not part of the lineage.
* A config change (whisper model, target LUFS, ...) invalidates the stage
  even when no file hash changed.
* Downstream stages inherit staleness transitively (BFS to fixed point).
* CLI overrides can force a stage range, or force everything.

The output is a topologically-sorted list of stage names ready for the
orchestrator to execute.
"""
from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable, Optional

from .manifest import Manifest

# Canonical pipeline order. Stages must be executed in this order; the
# invalidation algorithm uses this list to topologically sort the stale set
# and to interpret ``--from-stage`` / ``--to-stage`` ranges.
STAGE_ORDER: list[str] = [
    "extract",
    "separate",
    "diarize",
    "samples",
    "transcribe",
    "translate",
    "generate",
    "align",
    "reconstruct",
    "mix",
    "video",
]


@dataclass(frozen=True)
class CliOverrides:
    """User-supplied overrides from the ``generate`` subcommand."""

    force: bool = False
    from_stage: Optional[str] = None
    to_stage: Optional[str] = None


def _is_editable(path: str, manifest: Manifest) -> bool:
    """Return True if ``path`` is a user-editable artefact (spec §9)."""
    for pat in manifest.editable_paths:
        if "*" in pat:
            if fnmatch(path, pat):
                return True
        elif pat == path:
            return True
    return False


def _is_derived(path: str, manifest: Manifest) -> bool:
    """Return True if ``path`` is a derived artefact (regenerated as a side-effect)."""
    for pat in manifest.derived_paths:
        if "*" in pat:
            if fnmatch(path, pat):
                return True
        elif pat == path:
            return True
    return False


def _stage_has_config_change(
    stage_name: str,
    manifest: Manifest,
    current_configs: Optional[dict[str, dict]],
) -> bool:
    """Return True if ``stage``'s config differs from the recorded config.

    A config change (whisper model, target LUFS, ...) invalidates the stage
    even when no file hash changed. ``current_configs`` is the dict of
    currently requested settings; if it's ``None`` we conservatively
    assume no config change.

    ``current_configs`` is built by :meth:`WorkspacePipeline.generate` via
    ``getattr(self, k, None)`` for every field in the stage's
    ``config_fields``.  Fields that the pipeline does not carry (e.g.
    ``separate``'s ``model``/``device``/``out_dir_name`` — Demucs
    settings that are not exposed on the CLI) come back as ``None``.
    A ``None`` current value is not a user override; treating it as
    different from the recorded value would spuriously re-run an
    already-finished stage.  Only explicit non-``None`` values count.
    """
    if current_configs is None:
        return False

    # If the stage is not in current_configs, it means we don't have
    # new settings for it (e.g. it's a stage from a different phase).
    if stage_name not in current_configs:
        return False

    current = current_configs[stage_name]

    # Get the recorded config from the manifest.
    rec = manifest.stages.get(stage_name)
    if rec is None:
        return True  # Brand new stage

    recorded = rec.config

    # A field is "changed" only when the current value is not None AND
    # differs from what the stage recorded last time.  None entries
    # represent fields the pipeline does not track; they are gaps, not
    # overrides.
    for key, current_val in current.items():
        if current_val is None:
            continue
        if current_val != recorded.get(key):
            return True
    return False


def _iter_io(manifest: Manifest) -> Iterable[tuple[str, str, str]]:
    """Yield ``(stage_name, "in"|"out", path)`` for every recorded I/O entry."""
    for stage_name, stage in manifest.stages.items():
        for ref in stage.inputs:
            yield stage_name, "in", ref.path
        for ref in stage.outputs:
            yield stage_name, "out", ref.path


def _recorded_io(manifest: Manifest) -> dict[str, dict[str, str]]:
    """Return a ``{path: {"sha256": ..., "stage": ...}}`` map of all recorded I/O."""
    out: dict[str, dict[str, str]] = {}
    for stage_name, stage in manifest.stages.items():
        for ref in stage.inputs + stage.outputs:
            out[ref.path] = {"sha256": ref.sha256, "stage": stage_name}
    return out


def compute_stale_set(
    manifest: Manifest,
    workspace_root: Path,
    cli_overrides: Optional[CliOverrides] = None,
    *,
    current_configs: Optional[dict[str, dict]] = None,
) -> list[str]:
    """Return the topologically-sorted list of stages that need to re-run.

    Implements the 5-step algorithm from spec §10:

    1. Recompute SHA-256 for every recorded path that exists on disk.
    2. Detect hash and config mismatches per stage; mark producers (for
       non-editable inputs) or just the consumer (for editable inputs) as
       stale.
    3. Propagate downstream: any stage whose non-derived input is produced
       by a stale stage is itself stale. Repeat to fixed point.
    4. Apply CLI overrides (force / from_stage / to_stage).
    5. Return the result in :data:`STAGE_ORDER` order.
    """
    from .atomic import sha256_file  # local import to avoid cycle at module load

    cli_overrides = cli_overrides or CliOverrides()
    workspace_root = Path(workspace_root)

    recorded = _recorded_io(manifest)

    # 1. Recompute hashes for paths that exist on disk.
    actual_hashes: dict[str, Optional[str]] = {}
    for path in recorded:
        p = workspace_root / path
        if p.exists() and p.is_file():
            actual_hashes[path] = sha256_file(p)
        else:
            actual_hashes[path] = None

    # 2. Detect hash-mismatches and config-mismatches.
    stale: set[str] = set()
    for stage_name, stage in manifest.stages.items():
        for ref in stage.inputs:
            recorded_hash = ref.sha256
            current_hash = actual_hashes.get(ref.path)
            if current_hash is None or current_hash != recorded_hash:
                if _is_editable(ref.path, manifest):
                    # Trust the user's edit: do not walk upstream.
                    stale.add(stage_name)
                else:
                    producer = manifest.find_producer(ref.path)
                    if producer is not None:
                        stale.add(producer)
                    stale.add(stage_name)
        if _stage_has_config_change(stage_name, manifest, current_configs):
            stale.add(stage_name)

    # 3. Propagate downstream until fixed point.
    changed = True
    while changed:
        changed = False
        for stage_name, stage in manifest.stages.items():
            if stage_name in stale:
                continue
            for ref in stage.inputs:
                if _is_derived(ref.path, manifest):
                    continue
                producer = manifest.find_producer(ref.path)
                if producer in stale and stage_name not in stale:
                    stale.add(stage_name)
                    changed = True

    # 4. CLI overrides.
    if cli_overrides.force:
        stale = set(manifest.stages.keys())
    if cli_overrides.from_stage:
        from_idx = STAGE_ORDER.index(cli_overrides.from_stage)
        for s in STAGE_ORDER[from_idx:]:
            stale.add(s)
    if cli_overrides.to_stage:
        to_idx = STAGE_ORDER.index(cli_overrides.to_stage)
        stale = {s for s in stale if STAGE_ORDER.index(s) <= to_idx}

    # 5. Sort topologically.
    return [s for s in STAGE_ORDER if s in stale]
