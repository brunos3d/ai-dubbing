"""Workspace path utilities: slug, root resolution, and ID generation/parsing.

Spec section 7 defines the workspace ID layout::

    <source-slug>-<YYYYMMDD>-<hash8>/

where:

* ``source-slug`` is the input filename stem, slugified.
* ``YYYYMMDD`` is the local date of ``prepare``.
* ``hash8`` is the first 8 hex chars of ``content_hash``.

``content_hash`` is ``sha256(media_sha256 || src_lang || tgt_lang || pipeline_config_hash)``
(the ``hf_token`` is intentionally never part of the hash).
"""
from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional


class WorkspacePathError(ValueError):
    """Raised when a workspace path or ID is malformed."""


def workspaces_root() -> Path:
    """Return the absolute path to the workspaces root directory.

    Resolution order (highest priority first):

    1. ``AI_DUBBING_WORKSPACES_ROOT`` (literal override, no suffix appended).
    2. ``XDG_DATA_HOME/ai-dubbing/workspaces``.
    3. ``$HOME/.local/share/ai-dubbing/workspaces``.
    """
    override = os.environ.get("AI_DUBBING_WORKSPACES_ROOT")
    if override:
        return Path(override)

    xdg_data = os.environ.get("XDG_DATA_HOME")
    if xdg_data:
        return Path(xdg_data) / "ai-dubbing" / "workspaces"

    home = os.environ.get("HOME", "")
    return Path(home) / ".local" / "share" / "ai-dubbing" / "workspaces"


def slugify(name: str) -> str:
    """Lowercase ``name``; collapse runs of non-alphanumerics into ``-``; strip edges."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def workspace_id(
    media_sha256: str,
    src_lang: str,
    tgt_lang: str,
    pipeline_config_hash: str,
    source_slug: str,
    date_yyyymmdd: Optional[str] = None,
) -> str:
    """Build a workspace ID of the form ``<slug>-<YYYYMMDD>-<hash8>``."""
    content_hash = hashlib.sha256(
        f"{media_sha256}{src_lang}{tgt_lang}{pipeline_config_hash}".encode()
    ).hexdigest()

    slug = slugify(source_slug) or "workspace"
    date = date_yyyymmdd or datetime.now().strftime("%Y%m%d")
    return f"{slug}-{date}-{content_hash[:8]}"


_WORKSPACE_ID_RE = re.compile(
    r"^(?P<slug>.+)-(?P<date>\d{8})-(?P<hash>[0-9a-f]{8})$"
)


def parse_workspace_id(s: str) -> dict:
    """Parse a workspace ID; raise :class:`WorkspacePathError` on no match.

    Returns a dict with keys ``slug``, ``date``, ``hash``.
    """
    m = _WORKSPACE_ID_RE.match(s)
    if not m:
        raise WorkspacePathError(f"Invalid workspace ID: {s!r}")
    return {
        "slug": m.group("slug"),
        "date": m.group("date"),
        "hash": m.group("hash"),
    }
