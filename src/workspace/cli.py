"""Workspace subcommand handlers (Task 11).

The workspace CLI is a thin command dispatcher used by ``src.cli``'s
``workspace`` subparser. It is intentionally dependency-free (no
``argparse``, no I/O beyond the workspace root) so it can be tested
with simple ``argparse.Namespace``-like objects.

Subcommands:

* ``list`` — print a table of all workspaces.
* ``inspect`` — print the manifest's stage table.
* ``show`` — print the path of a subdirectory of the workspace.
* ``validate`` — run every per-artifact validator; return 0 on success, 2
  if any error-severity issue was found.
* ``clean`` — delete a workspace (or only its ``output/``) after a
  confirmation prompt.
* ``open`` — print the workspace root path.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Optional

from .manifest import Manifest
from .paths import parse_workspace_id, workspaces_root
from .validate import (
    Issue,
    validate_diarization_segments,
    validate_glossary,
    validate_manifest,
    validate_metadata,
    validate_transcript,
    validate_translation,
)

STANDARD_SUBDIRS: tuple[str, ...] = (
    "media",
    "diarization",
    "transcription",
    "translation",
    "speakers",
    "output",
    "source",
    "logs",
)


def _workspaces() -> Path:
    """Return the configured workspaces root."""
    return workspaces_root()


def _all_workspaces() -> list[Path]:
    """Return all workspace directories, sorted by name, that actually exist."""
    root = _workspaces()
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def _try_load(path: Path) -> dict:
    """Silently load JSON from ``path``; return ``{}`` on any failure."""
    try:
        loaded = json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _gather_issues(root: Path) -> list[Issue]:
    """Run every per-artifact validator on the files that exist in ``root``.

    Files that don't exist are skipped (the workspace may simply not have
    reached that stage yet). Files that exist but are unreadable/broken
    yield no issues either — broken JSON is loaded as ``{}`` and the
    validator reports its own findings, but we still go ahead so a single
    bad file doesn't crash the whole check.
    """
    issues: list[Issue] = []
    candidates: tuple[tuple[str, callable], ...] = (
        ("metadata.json", validate_metadata),
        ("manifest.json", validate_manifest),
        ("diarization/segments.json", validate_diarization_segments),
        ("transcription/transcript.json", validate_transcript),
        ("translation/translated_transcript.json", validate_translation),
        ("translation/glossary.json", validate_glossary),
    )
    for relpath, validator in candidates:
        p = Path(root) / relpath
        if not p.exists():
            continue
        data = _try_load(p)
        issues.extend(validator(data))
    return issues


def _resolve_workspace(workspace_id: str) -> Path:
    """Return the on-disk path for ``workspace_id``; raise if missing."""
    try:
        parse_workspace_id(workspace_id)
    except Exception as exc:  # noqa: BLE001 - rewrapped below
        raise SystemExit(f"invalid workspace id {workspace_id!r}: {exc}")
    root = _workspaces() / workspace_id
    if not root.exists():
        raise SystemExit(f"workspace {workspace_id!r} not found at {root}")
    return root


def cmd_workspace_list(args: Optional[object] = None) -> int:
    """Print a table of all workspaces; return 0 always."""
    workspaces = _all_workspaces()
    if not workspaces:
        print("No workspaces found.")
        return 0

    rows: list[tuple[str, str, str, str, str]] = []
    for ws in workspaces:
        meta = _try_load(ws / "metadata.json")
        source = meta.get("source") if isinstance(meta, dict) else None
        src = source.get("source_language", "?") if isinstance(source, dict) else "?"
        tgt = source.get("target_language", "?") if isinstance(source, dict) else "?"
        created = meta.get("created_at", "?") if isinstance(meta, dict) else "?"
        media_path = source.get("media_path", "?") if isinstance(source, dict) else "?"
        rows.append((ws.name, src, tgt, created, media_path))

    header = ("WORKSPACE ID", "SRC", "TGT", "CREATED", "SOURCE")
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(5)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    print(fmt.format(*("-" * w for w in widths)))
    for r in rows:
        print(fmt.format(*r))
    return 0


def cmd_workspace_inspect(args) -> int:
    """Print the manifest's stage table for the given workspace."""
    root = _resolve_workspace(args.workspace_id)
    manifest = Manifest.load(root / "manifest.json")
    if manifest is None:
        print(f"No manifest found at {root / 'manifest.json'}.")
        return 0

    print(f"Workspace: {args.workspace_id}")
    print(f"Path:      {root}")
    print(f"Pipeline:  {manifest.pipeline_version} (commit {manifest.git_commit})")
    print()
    print(f"{'STAGE':<14}  {'STATUS':<10}  {'DURATION (s)':>12}")
    print(f"{'-' * 14}  {'-' * 10}  {'-' * 12}")
    for name in manifest.stages:
        rec = manifest.stages[name]
        duration = f"{rec.duration_s:.2f}" if rec.duration_s is not None else "-"
        print(f"{name:<14}  {rec.status:<10}  {duration:>12}")
    return 0


def cmd_workspace_show(args) -> int:
    """Print the path of a subdir of the workspace, or list all standard subdirs."""
    root = _resolve_workspace(args.workspace_id)
    subdir: Optional[str] = getattr(args, "path", None)
    if subdir:
        target = root / subdir
        print(target)
        return 0
    print(f"Workspace: {root}")
    for name in STANDARD_SUBDIRS:
        print(f"  {root / name}")
    return 0


def cmd_workspace_validate(args) -> int:
    """Run every validator on the workspace; return 0 on success, 2 on errors."""
    root = _resolve_workspace(args.workspace_id)
    issues = _gather_issues(root)
    if not issues:
        print(f"Workspace {args.workspace_id}: no issues found.")
        return 0
    for issue in issues:
        print(f"[{issue.severity}] {issue.path}: {issue.message}")
    errors = sum(1 for i in issues if i.severity == "error")
    if errors:
        print(f"\n{errors} error(s), {len(issues) - errors} warning(s).")
        return 2
    print(f"\n{len(issues)} warning(s); no errors.")
    return 0


def cmd_workspace_clean(args) -> int:
    """Delete a workspace (or only its ``output/``); return 0 on success."""
    root = _resolve_workspace(args.workspace_id)
    keep_outputs = bool(getattr(args, "keep_outputs", False))
    yes = bool(getattr(args, "yes", False))

    if keep_outputs:
        target = root / "output"
        action = f"remove {target}"
    else:
        target = root
        action = f"remove entire workspace {root}"

    if not yes:
        try:
            reply = input(f"{action}? [y/N] ").strip().lower()
        except EOFError:
            reply = ""
        if reply not in {"y", "yes"}:
            print("Aborted.")
            return 1

    if not target.exists():
        print(f"Nothing to remove at {target}.")
        return 0

    shutil.rmtree(target)
    print(f"Removed {target}.")
    return 0


def cmd_workspace_open(args) -> int:
    """Print the workspace root path."""
    root = _resolve_workspace(args.workspace_id)
    print(root)
    return 0
