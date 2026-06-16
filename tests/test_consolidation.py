"""Phase-1A consolidation tests (spec 2026-06-16).

Covers the behaviour that lets ``WorkspacePipeline`` be the single
orchestrator:

* ``prepare()`` is now staleness-aware — a second ``prepare()`` of an
  unchanged input re-runs *zero* stages (cache hit), where the old
  unconditional ``prepare()`` re-ran all six analysis stages.
* ``run_oneshot()`` runs prepare + generate and delivers the final
  artefacts into the requested ``output_dir`` (the ``dub.sh`` contract).
* The legacy ``Pipeline`` shim still exposes ``pipeline_config_dict()``.

The heavy stage classes are replaced with the in-process stubs from
``tests/test_workspace_e2e.py`` (GPU-free).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_workspace_e2e import install_stubs  # noqa: E402


@pytest.fixture
def fixture_audio() -> str:
    p = Path(__file__).resolve().parent / "fixtures" / "short_sample.wav"
    if not p.exists():
        import subprocess

        subprocess.check_call(
            ["python", str(p.parent / "generate_short_sample.py")]
        )
    return str(p)


def test_prepare_is_idempotent(monkeypatch, tmp_path, fixture_audio) -> None:
    """A second prepare() of an unchanged input re-runs nothing."""
    from src.workspace.pipeline import WorkspacePipeline

    monkeypatch.setenv("AI_DUBBING_WORKSPACES_ROOT", str(tmp_path))
    record: List[str] = []
    install_stubs(monkeypatch, record)

    wsp = WorkspacePipeline(
        input_path=fixture_audio, source_language="en", target_language="es"
    )
    wsp.prepare()
    assert record == [
        "extract", "separate", "diarize", "samples", "transcribe", "translate"
    ], record

    # Second prepare: everything is cached, nothing should re-run.
    record.clear()
    wsp2 = WorkspacePipeline(
        input_path=fixture_audio, source_language="en", target_language="es"
    )
    wsp2.prepare()
    assert record == [], f"prepare() re-ran stages that were already fresh: {record}"


def test_run_oneshot_delivers_outputs(monkeypatch, tmp_path, fixture_audio) -> None:
    """run_oneshot() runs prepare+generate and copies finals into output_dir."""
    from src.workspace.pipeline import WorkspacePipeline

    monkeypatch.setenv("AI_DUBBING_WORKSPACES_ROOT", str(tmp_path / "ws"))
    record: List[str] = []
    install_stubs(monkeypatch, record)

    out_dir = tmp_path / "delivered"
    wsp = WorkspacePipeline(
        input_path=fixture_audio, source_language="en", target_language="es"
    )
    result = wsp.run_oneshot(output_dir=out_dir, skip_video=True)

    # prepare + generate (minus video, capped at mix) all ran.
    assert "generate" in record and "mix" in record
    assert "video" not in record, "skip_video should cap the run at mix"
    # Final audio delivered into the requested output dir.
    assert (out_dir / "final_audio.wav").exists(), result


def test_pipeline_shim_config_dict(tmp_path) -> None:
    """The legacy Pipeline shim still exposes a token-free config dict."""
    from src.pipeline import Pipeline

    f = tmp_path / "input.bin"
    f.write_bytes(b"x")
    p = Pipeline(
        input_path=str(f),
        source_language="en",
        target_language="es",
        workdir=tmp_path,
        hf_token="secret",
        no_pyannote=True,
    )
    cfg = p.pipeline_config_dict()
    assert "hf_token" not in cfg
    assert cfg["hf_token_available"] is True
    assert cfg["no_pyannote"] is True
