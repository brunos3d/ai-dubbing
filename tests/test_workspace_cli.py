"""Tests for ``src/workspace/cli.py`` (Task 11).

The CLI handlers (list / inspect / show / validate / clean / open) are
exercised against a seeded workspace rooted at ``$AI_DUBBING_WORKSPACES_ROOT``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.workspace import cli  # noqa: E402


def _ns(**kwargs) -> argparse.Namespace:
    """Build an argparse.Namespace with only the requested fields set."""
    return SimpleNamespace(**kwargs)


def _seed_workspace(workspaces: Path) -> str:
    """Materialise a minimal workspace and return its id.

    Writes ``manifest.json`` (with an ``extract`` stage that has the
    ``media/original_audio.wav`` output), ``metadata.json`` (matching the
    :func:`validate_metadata` schema), and the audio file itself.
    """
    wid = "demo-20260613-deadbeef"
    root = workspaces / wid
    (root / "media").mkdir(parents=True)

    wav_path = root / "media" / "original_audio.wav"
    sf.write(str(wav_path), __import__("numpy").zeros(16000, dtype="float32"), 16000)

    manifest = {
        "schema_version": 1,
        "workspace_id": wid,
        "pipeline_version": "0.1.0",
        "git_commit": "abc1234",
        "editable_paths": [],
        "derived_paths": [],
        "stages": {
            "extract": {
                "name": "extract",
                "status": "done",
                "config": {"sample_rate": 16000},
                "started_at": "2026-06-13T12:00:00Z",
                "finished_at": "2026-06-13T12:00:01Z",
                "duration_s": 1.0,
                "inputs": [],
                "outputs": [
                    {
                        "path": "media/original_audio.wav",
                        "sha256": __import__("hashlib").sha256(
                            wav_path.read_bytes()
                        ).hexdigest(),
                        "size_bytes": wav_path.stat().st_size,
                    }
                ],
                "error": None,
            }
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    metadata = {
        "workspace_schema_version": 1,
        "workspace_id": wid,
        "content_hash": "deadbeef",
        "created_at": "2026-06-13T12:00:00Z",
        "updated_at": "2026-06-13T12:00:00Z",
        "workspace_created_with": {
            "pipeline_version": "0.1.0",
            "git_commit": "abc1234",
            "python_version": "3.12.4",
        },
        "source": {
            "media_path": "/tmp/demo.wav",
            "media_sha256": "a" * 64,
            "source_language": "pt",
            "target_language": "en",
        },
        "config": {
            "whisper_model": "large-v3",
            "hf_token_available": False,
            "min_speakers": None,
            "max_speakers": None,
            "no_pyannote": False,
            "target_lufs": -16.0,
            "glossary_path": None,
        },
        "pipeline_config_hash": "b" * 64,
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return wid


def test_workspace_list(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setenv("AI_DUBBING_WORKSPACES_ROOT", str(tmp_path))
    wid = _seed_workspace(tmp_path)
    rc = cli.cmd_workspace_list(None)
    out = capsys.readouterr().out
    assert rc == 0
    assert wid in out


def test_workspace_inspect(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setenv("AI_DUBBING_WORKSPACES_ROOT", str(tmp_path))
    wid = _seed_workspace(tmp_path)
    rc = cli.cmd_workspace_inspect(_ns(workspace_id=wid))
    out = capsys.readouterr().out
    assert rc == 0
    assert "extract" in out


def test_workspace_show_prints_path(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setenv("AI_DUBBING_WORKSPACES_ROOT", str(tmp_path))
    wid = _seed_workspace(tmp_path)
    rc = cli.cmd_workspace_show(_ns(workspace_id=wid))
    out = capsys.readouterr().out
    assert rc == 0
    assert "media" in out


def test_workspace_validate_clean(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setenv("AI_DUBBING_WORKSPACES_ROOT", str(tmp_path))
    wid = _seed_workspace(tmp_path)
    rc = cli.cmd_workspace_validate(_ns(workspace_id=wid))
    out = capsys.readouterr().out
    assert rc == 0
    assert "no issues" in out


def test_workspace_clean_removes(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setenv("AI_DUBBING_WORKSPACES_ROOT", str(tmp_path))
    wid = _seed_workspace(tmp_path)
    root = tmp_path / wid
    assert root.exists()
    rc = cli.cmd_workspace_clean(_ns(workspace_id=wid, keep_outputs=False, yes=True))
    assert rc == 0
    assert not root.exists()
