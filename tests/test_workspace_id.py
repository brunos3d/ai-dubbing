"""Tests for ``src.workspace.content_hash``.

Covers:

* ``media_sha256`` matches stdlib ``hashlib.sha256`` on a known input.
* ``content_hash`` excludes the ``hf_token`` keyword.
* ``content_hash`` changes when any of its four real inputs change.
* ``pipeline_config_hash`` is deterministic, key-order invariant, value
  sensitive, supports nested dicts, and ignores the ``hf_token`` key.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.workspace.content_hash import (  # noqa: E402
    content_hash,
    media_sha256,
    pipeline_config_hash,
)


MEDIA = "a" * 64
CONFIG = "b" * 64


def test_media_sha256_matches_stdlib(tmp_path: Path) -> None:
    p = tmp_path / "blob.bin"
    p.write_bytes(b"hello world")
    assert media_sha256(p) == hashlib.sha256(b"hello world").hexdigest()


def test_content_hash_excludes_hf_token() -> None:
    h_secret = content_hash(
        media_sha256=MEDIA,
        src_lang="en",
        tgt_lang="es",
        pipeline_config_hash=CONFIG,
        hf_token="secret",
    )
    h_different = content_hash(
        media_sha256=MEDIA,
        src_lang="en",
        tgt_lang="es",
        pipeline_config_hash=CONFIG,
        hf_token="different",
    )
    assert h_secret == h_different


def test_content_hash_changes_with_inputs() -> None:
    baseline = content_hash(
        media_sha256=MEDIA,
        src_lang="en",
        tgt_lang="es",
        pipeline_config_hash=CONFIG,
    )
    variants = {
        "media": content_hash(
            media_sha256="c" * 64,
            src_lang="en",
            tgt_lang="es",
            pipeline_config_hash=CONFIG,
        ),
        "src": content_hash(
            media_sha256=MEDIA,
            src_lang="pt",
            tgt_lang="es",
            pipeline_config_hash=CONFIG,
        ),
        "tgt": content_hash(
            media_sha256=MEDIA,
            src_lang="en",
            tgt_lang="fr",
            pipeline_config_hash=CONFIG,
        ),
        "config": content_hash(
            media_sha256=MEDIA,
            src_lang="en",
            tgt_lang="es",
            pipeline_config_hash="d" * 64,
        ),
        "baseline_again": content_hash(
            media_sha256=MEDIA,
            src_lang="en",
            tgt_lang="es",
            pipeline_config_hash=CONFIG,
        ),
    }
    distinct = {baseline, variants["media"], variants["src"], variants["tgt"], variants["config"]}
    assert len(distinct) == 5
    assert baseline == variants["baseline_again"]


def test_pipeline_config_hash_is_deterministic() -> None:
    cfg = {"whisper_model": "large-v3", "target_lufs": -16.0, "no_pyannote": False}
    assert pipeline_config_hash(cfg) == pipeline_config_hash(cfg)


def test_pipeline_config_hash_key_order_invariant() -> None:
    a = {"a": 1, "b": 2, "c": 3}
    b = {"c": 3, "b": 2, "a": 1}
    assert pipeline_config_hash(a) == pipeline_config_hash(b)


def test_pipeline_config_hash_changes_with_value() -> None:
    assert pipeline_config_hash({"a": 1}) != pipeline_config_hash({"a": 2})


def test_pipeline_config_hash_nested_dicts() -> None:
    a = {"outer": {"a": 1, "b": 2}}
    b = {"outer": {"b": 2, "a": 1}}
    assert pipeline_config_hash(a) == pipeline_config_hash(b)


def test_pipeline_config_hash_ignores_hf_token() -> None:
    secret = {"whisper_model": "large-v3", "hf_token": "secret"}
    different = {"whisper_model": "large-v3", "hf_token": "different"}
    assert pipeline_config_hash(secret) == pipeline_config_hash(different)
