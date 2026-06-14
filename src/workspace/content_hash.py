"""Content-derived hashes for workspace identity.

These hashes are the inputs to the workspace ID, so they must be:

* **Deterministic** — same inputs always produce the same hash.
* **Key-order invariant** — JSON reordering of dicts does not change the hash.
* **Independent of secrets** — ``hf_token`` and other auth material must not
  influence identity (rotating a token must not invalidate workspaces).

Three functions are exposed:

* :func:`media_sha256` — SHA-256 of a media file, streamed in 64 KB chunks so
  arbitrarily large inputs can be hashed in constant memory.
* :func:`pipeline_config_hash` — SHA-256 of a deterministic JSON dump of a
  pipeline configuration dict, with secret-bearing keys stripped.
* :func:`content_hash` — SHA-256 of the concatenation of the media hash, the
  source/target language codes, and the pipeline config hash. The
  ``hf_token`` parameter is accepted for API symmetry but never participates
  in the hash.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

_HASH_EXCLUDE_KEYS: frozenset[str] = frozenset(
    {"hf_token", "hf_token_available", "offload_dir"}
)

_CHUNK_SIZE: int = 64 * 1024


def media_sha256(path: str | Path) -> str:
    """Return the hex SHA-256 of ``path``, streamed in 64 KB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def _strip_excluded(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return a new dict with ``_HASH_EXCLUDE_KEYS`` removed recursively.

    Nested dicts are walked so that a secret in a sub-config (e.g.
    ``{"translation": {"hf_token": "..."}}``) is also stripped. Lists are
    preserved as-is; only dicts are inspected.
    """
    result: dict[str, Any] = {}
    for key, value in cfg.items():
        if key in _HASH_EXCLUDE_KEYS:
            continue
        if isinstance(value, Mapping):
            result[key] = _strip_excluded(value)
        else:
            result[key] = value
    return result


def pipeline_config_hash(cfg: Mapping[str, Any]) -> str:
    """Return the hex SHA-256 of a deterministic JSON dump of ``cfg``.

    Excluded keys (see :data:`_HASH_EXCLUDE_KEYS`) are stripped first. The
    JSON is dumped with ``sort_keys=True`` and the compact ``(",", ":")``
    separator so that dict ordering does not change the hash. ``default=str``
    makes the dump tolerant of non-JSON-native values such as ``Path``.
    """
    stripped = _strip_excluded(cfg)
    payload = json.dumps(stripped, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def content_hash(
    *,
    media_sha256: str,
    src_lang: str,
    tgt_lang: str,
    pipeline_config_hash: str,
    hf_token: str | None = None,
) -> str:
    """Return the hex SHA-256 of the workspace identity inputs.

    The hash is ``sha256(media_sha256 || src_lang || tgt_lang ||
    pipeline_config_hash)``. The ``hf_token`` keyword is accepted for caller
    convenience and is intentionally ignored — token rotation must not
    invalidate existing workspaces.
    """
    del hf_token  # accepted but never part of the hash
    payload = f"{media_sha256}{src_lang}{tgt_lang}{pipeline_config_hash}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
