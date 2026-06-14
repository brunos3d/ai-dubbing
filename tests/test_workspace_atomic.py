"""Tests for src/workspace/atomic.py (Task 5)."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.workspace.atomic import (  # noqa: E402
    AtomicWriteError,
    promote,
    sha256_file,
    stage_staging_dir,
)


def test_stage_staging_dir_lives_under_dot_tmp(tmp_path: Path) -> None:
    p = stage_staging_dir(tmp_path, "extract")
    assert p.parent == tmp_path / ".tmp"
    assert p.name.startswith("extract-")
    suffix = p.name[len("extract-"):]
    assert len(suffix) == 8


def test_promote_moves_files_atomically(tmp_path: Path) -> None:
    staging = tmp_path / ".tmp" / "extract-abcdef01"
    staging.mkdir(parents=True)
    (staging / "a.txt").write_text("alpha")
    (staging / "sub").mkdir()
    (staging / "sub" / "b.txt").write_text("beta")
    promote(staging, tmp_path)
    assert (tmp_path / "a.txt").read_text() == "alpha"
    assert (tmp_path / "sub" / "b.txt").read_text() == "beta"
    assert not staging.exists()


def test_promote_no_op_when_staging_missing(tmp_path: Path) -> None:
    promote(tmp_path / "no-such-dir", tmp_path)  # must not raise


def test_promote_preserves_existing_files(tmp_path: Path) -> None:
    (tmp_path / "keep.txt").write_text("original")
    staging = tmp_path / ".tmp" / "extract-abcdef01"
    staging.mkdir(parents=True)
    (staging / "new.txt").write_text("fresh")
    promote(staging, tmp_path)
    assert (tmp_path / "keep.txt").read_text() == "original"
    assert (tmp_path / "new.txt").read_text() == "fresh"


def test_sha256_file(tmp_path: Path) -> None:
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello")
    assert sha256_file(p) == hashlib.sha256(b"hello").hexdigest()


def test_promote_cleans_staging_on_failure(tmp_path: Path) -> None:
    # Block mkdir by putting a file at tmp_path/sub where a directory is needed.
    (tmp_path / "sub").write_text("blocker")
    staging = tmp_path / ".tmp" / "extract-abcdef01"
    staging.mkdir(parents=True)
    (staging / "sub").mkdir()
    (staging / "sub" / "b.txt").write_text("beta")
    with pytest.raises(AtomicWriteError):
        promote(staging, tmp_path)
    # The .tmp staging dir must be cleaned up.
    assert not staging.exists()
