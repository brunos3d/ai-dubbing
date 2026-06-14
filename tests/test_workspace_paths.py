"""Tests for :mod:`src.workspace.paths`."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make `src` importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.workspace.paths import (  # noqa: E402
    WorkspacePathError,
    parse_workspace_id,
    slugify,
    workspace_id,
    workspaces_root,
)


def test_workspaces_root_default_is_local_share(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.delenv("AI_DUBBING_WORKSPACES_ROOT", raising=False)
    assert workspaces_root() == tmp_path / ".local" / "share" / "ai-dubbing" / "workspaces"


def test_workspaces_root_respects_xdg_data_home(monkeypatch, tmp_path):
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg))
    monkeypatch.delenv("AI_DUBBING_WORKSPACES_ROOT", raising=False)
    assert workspaces_root() == xdg / "ai-dubbing" / "workspaces"


def test_workspaces_root_override(monkeypatch, tmp_path):
    override = tmp_path / "myroot"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setenv("AI_DUBBING_WORKSPACES_ROOT", str(override))
    assert workspaces_root() == override


def test_slugify_lowercases_and_replaces_non_alnum():
    assert slugify("Peter Ei-Nerd 2026!!!") == "peter-ei-nerd-2026"
    assert slugify("Hello World") == "hello-world"
    assert slugify("___") == ""


def test_workspace_id_format():
    wid = workspace_id(
        media_sha256="a" * 64,
        src_lang="pt",
        tgt_lang="en",
        pipeline_config_hash="b" * 64,
        source_slug="my-clip",
        date_yyyymmdd="20260613",
    )
    parts = wid.split("-")
    assert parts[-2] == "20260613"
    assert len(parts[-1]) == 8
    assert all(c in "0123456789abcdef" for c in parts[-1])
    assert "-".join(parts[:-2]) == "my-clip"


def test_workspace_id_deterministic():
    kwargs = dict(
        media_sha256="a" * 64,
        src_lang="pt",
        tgt_lang="en",
        pipeline_config_hash="b" * 64,
        source_slug="my-clip",
        date_yyyymmdd="20260613",
    )
    assert workspace_id(**kwargs) == workspace_id(**kwargs)


def test_workspace_id_changes_with_config_hash():
    common = dict(
        media_sha256="a" * 64,
        src_lang="pt",
        tgt_lang="en",
        source_slug="my-clip",
        date_yyyymmdd="20260613",
    )
    wid1 = workspace_id(pipeline_config_hash="b" * 64, **common)
    wid2 = workspace_id(pipeline_config_hash="c" * 64, **common)
    assert wid1 != wid2


def test_parse_workspace_id_round_trip():
    wid = workspace_id(
        media_sha256="a" * 64,
        src_lang="pt",
        tgt_lang="en",
        pipeline_config_hash="b" * 64,
        source_slug="my-clip",
        date_yyyymmdd="20260613",
    )
    parsed = parse_workspace_id(wid)
    assert parsed["slug"] == "my-clip"
    assert parsed["date"] == "20260613"
    assert len(parsed["hash"]) == 8


def test_parse_workspace_id_rejects_garbage():
    with pytest.raises(WorkspacePathError):
        parse_workspace_id("not-a-valid-id")
    with pytest.raises(WorkspacePathError):
        parse_workspace_id("slug-20260101-badhex!@")
