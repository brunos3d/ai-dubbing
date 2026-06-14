"""Atomic stage writes for the workspace pipeline (spec §11).

Each stage runs against a temporary staging directory and only on success are
its outputs *promoted* into the real workspace. This guarantees that a crash or
exception mid-stage leaves the workspace unchanged — the next invocation will
re-run from the same point.

The contract:

* :func:`stage_staging_dir` returns a unique path under ``<root>/.tmp/``.
* :func:`sha256_file` computes a SHA-256 of a file's bytes (used to fill
  ``ArtifactRef`` records after a successful promote).
* :func:`promote` moves everything in the staging dir into the workspace root
  in one shot and removes the staging dir. On any failure it removes the
  staging dir and raises :class:`AtomicWriteError` so the caller can record
  a failed stage without polluting the workspace.
"""
from __future__ import annotations

import hashlib
import shutil
import uuid
from pathlib import Path

_CHUNK_SIZE: int = 64 * 1024


class AtomicWriteError(RuntimeError):
    """Raised when a stage's outputs cannot be promoted into the workspace."""


def stage_staging_dir(root: Path, stage_name: str) -> Path:
    """Return a unique staging dir under ``<root>/.tmp/<stage_name>-<uuid8>``.

    The directory is *not* created here — the caller creates it before running
    the stage so a failure to create the dir is reported by the caller, not
    here.
    """
    suffix = uuid.uuid4().hex[:8]
    return Path(root) / ".tmp" / f"{stage_name}-{suffix}"


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of ``path``, streamed in 64 KB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def promote(staging: Path, root: Path) -> None:
    """Move everything in ``staging`` into ``root``; clean up on failure.

    Behaviour:

    * If ``staging`` does not exist this is a no-op.
    * On success, every entry under ``staging`` is moved into ``root`` (files
      and directories), preserving the relative layout, and ``staging`` is
      removed.
    * On any exception, ``staging`` is removed and :class:`AtomicWriteError`
      is raised, leaving the workspace untouched.
    """
    staging = Path(staging)
    root = Path(root)
    if not staging.exists():
        return
    try:
        for entry in staging.iterdir():
            dest = root / entry.name
            if entry.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                for child in entry.iterdir():
                    child_dest = dest / child.name
                    child_dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(child), str(child_dest))
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(entry), str(dest))
    except Exception as exc:  # noqa: BLE001 - we rewrap below
        shutil.rmtree(staging, ignore_errors=True)
        raise AtomicWriteError(
            f"failed to promote staging dir {staging} into {root}: {exc}"
        ) from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)
