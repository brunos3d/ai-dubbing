"""Tests for :mod:`src.workspace.pipeline` helpers (Task 10).

The five tests below cover the pure-helper surface of the new
``WorkspacePipeline`` module:

* ``stage_subdir`` — the per-stage workspace-relative subdir lookup.
* ``workspace_layout`` — the canonical list of subdirs a workspace owns.
* ``prepare_stages`` / ``generate_stages`` — the canonical stage ranges for
  the two-phase workflow.

No I/O, no real stages are executed here — these are the contract that the
heavier integration tests in ``tests/test_workspace_e2e.py`` and
``tests/test_workspace_edit_scenarios.py`` rely on.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make `src` importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.workspace.pipeline import (  # noqa: E402
    generate_stages,
    prepare_stages,
    stage_subdir,
    workspace_layout,
)


def test_stage_subdir():
    assert stage_subdir("extract") == "media"
    assert stage_subdir("diarize") == "diarization"
    assert stage_subdir("transcribe") == "transcription"
    assert stage_subdir("translate") == "translation"


def test_stage_subdir_root_for_synthesis_stages():
    for s in ("generate", "align", "reconstruct", "mix", "video"):
        assert stage_subdir(s) is None, f"{s} should be at workspace root"


def test_workspace_layout_lists_all_dirs():
    layout = workspace_layout()
    for d in (
        "media",
        "diarization",
        "transcription",
        "translation",
        "speakers",
        "output",
        "source",
        "logs",
    ):
        assert d in layout, f"missing {d!r} in workspace_layout()"


def test_prepare_stages_stops_at_translate():
    assert prepare_stages() == [
        "extract",
        "separate",
        "diarize",
        "samples",
        "transcribe",
        "translate",
    ]


def test_generate_stages_starts_at_generate():
    assert generate_stages() == [
        "generate",
        "align",
        "reconstruct",
        "mix",
        "video",
    ]


def test_speaker_profiles_loaded_from_disk_when_samples_not_re_run(tmp_path, monkeypatch):
    """If the generate stage is reached without the samples stage having
    run in the current session (e.g. on a ``dub.sh generate`` resume
    where samples is up-to-date), the orchestrator must still
    populate ``context["speaker_profiles"]`` from the
    ``speaker_profiles/`` directory on disk — otherwise the
    OmniVoice generate stage fails with
    ``"No speaker voice profiles available"``.

    The legacy ``Pipeline.run`` already does this (it walks
    ``speaker_profiles/*/primary/metadata.json`` and rebuilds the
    profiles dict).  The workspace pipeline must do the same.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.workspace import pipeline as wsp_mod

    # Materialise a minimal ``speaker_profiles/`` tree with one speaker.
    sp_dir = tmp_path / "speaker_profiles" / "speaker_01" / "primary"
    sp_dir.mkdir(parents=True)
    (sp_dir / "reference.wav").write_bytes(b"FAKE_WAV")
    (sp_dir / "transcript.txt").write_text("hello world", encoding="utf-8")
    (sp_dir / "metadata.json").write_text(json.dumps({
        "speaker_id": "speaker_01",
        "profile_id": "primary",
        "score": 99.5,
        "reference_duration": 1.5,
        "segments_used": 3,
        "transcript_words": 2,
        "reference_path": str(sp_dir / "reference.wav"),
        "transcript_path": str(sp_dir / "transcript.txt"),
    }))

    loaded = wsp_mod.load_speaker_profiles_from_disk(tmp_path)
    assert "speaker_01" in loaded, f"speaker_profiles not loaded: {loaded}"
    assert "primary" in loaded["speaker_01"]["profiles"]
    prof = loaded["speaker_01"]["profiles"]["primary"]
    assert prof["score"] == 99.5
    assert prof["transcript_text"] == "hello world"
    assert prof["reference"].endswith("reference.wav")


def test_stage_return_value_propagates_to_context(tmp_path, monkeypatch):
    """The workspace orchestrator must merge each stage's return value
    into the context so downstream stages can read the upstream
    artefacts (e.g. ``context["speaker_profiles"]`` set by the samples
    stage and consumed by the OmniVoice generate stage).

    Regression: the original ``_run_single_stage`` ignored the stage's
    return value entirely, which made the generate stage raise
    ``RuntimeError("No speaker voice profiles available")`` on every
    real workspace run.  The legacy ``Pipeline.run`` merges the return
    value into the context after every stage; the workspace pipeline
    must do the same.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.workspace import pipeline as wsp_mod

    captured = {}

    class StubSamplesStage:
        name = "samples"
        inputs = ["media/speech.wav", "diarization/segments.json"]
        outputs = ["speakers/speaker_01.wav"]
        editable_outputs = []
        derived_outputs = []
        config_fields = []

        def __init__(self, *args, **kwargs):
            pass

        def run(self, context):
            captured["saw_in_samples"] = dict(context)
            # Mimic what the real samples stage returns: the speaker
            # voice profiles dict that the generate stage reads.
            return {
                "speaker_profiles": {"speaker_01": {"profiles": {"primary": "REF"}}},
                "speaker_samples": {"speaker_01": "samples/speaker_01.wav"},
            }

    class StubGenerateStage:
        name = "generate"
        inputs = ["translation/translated_transcript.json"]
        outputs = ["generated_segments/manifest.json"]
        editable_outputs = []
        derived_outputs = ["generated_segments/manifest.json"]
        config_fields = ["model_id", "target_language", "use_clone_prompt"]

        def __init__(self, *args, **kwargs):
            pass

        def run(self, context):
            captured["saw_in_generate"] = dict(context)
            return {"generated_path": "generated_segments/manifest.json"}

    stages_run = [StubSamplesStage, StubGenerateStage]
    stage_name_iter = iter(["samples", "generate"])

    def fake_build(name, *args, **kwargs):
        cls = stages_run[0] if name == "samples" else stages_run[1]
        return cls()

    monkeypatch.setattr(wsp_mod, "STAGE_CLASSES", {
        "samples": StubSamplesStage,
        "generate": StubGenerateStage,
    })
    monkeypatch.setattr(wsp_mod, "_build_stage", fake_build)

    # Build a minimal manifest with both stages already seeded.
    from src.workspace.manifest import Manifest, StageRecord
    manifest = Manifest.create(
        workspace_id="t", pipeline_version="0.1.0", git_commit="x"
    )
    manifest.add_stage("samples", StageRecord(name="samples", status="pending"))
    manifest.add_stage("generate", StageRecord(name="generate", status="pending"))

    workspace_root = tmp_path / "ws"
    workspace_root.mkdir()

    wsp = wsp_mod.WorkspacePipeline(
        input_path="/dev/null",
        source_language="en",
        target_language="pt-BR",
        workspace_root=workspace_root,
    )
    # Inject pre-populated context keys so ``_run_stages`` does not bail
    # on missing well-known context keys.
    from src.workspace.pipeline import WorkspaceContext
    ctx = WorkspaceContext(
        workspace_root=workspace_root,
        input_path="/dev/null",
        source_language="en",
        target_language="pt-BR",
    )
    ctx["speech_path"] = str(workspace_root / "media" / "speech.wav")
    ctx["segments_path"] = str(workspace_root / "diarization" / "segments.json")
    ctx["translated_path"] = str(
        workspace_root / "translation" / "translated_transcript.json"
    )

    wsp._run_stages(
        manifest, ["samples", "generate"], start_at="samples",
        workspace_root=workspace_root,
        # Force the context to be ours (the function builds a new one)
        # by re-running through the helper; here we cheat: we call the
        # single-stage method directly.
    )

    # We bypass ``_run_stages`` (which rebuilds the context).  Run each
    # stage via ``_run_single_stage`` instead, propagating the context
    # by hand.
    manifest2 = Manifest.create(
        workspace_id="t", pipeline_version="0.1.0", git_commit="x"
    )
    manifest2.add_stage("samples", StageRecord(name="samples", status="pending"))
    manifest2.add_stage("generate", StageRecord(name="generate", status="pending"))
    ctx2 = WorkspaceContext(
        workspace_root=workspace_root,
        input_path="/dev/null",
        source_language="en",
        target_language="pt-BR",
    )
    ctx2["speech_path"] = str(workspace_root / "media" / "speech.wav")
    ctx2["segments_path"] = str(workspace_root / "diarization" / "segments.json")
    ctx2["translated_path"] = str(
        workspace_root / "translation" / "translated_transcript.json"
    )

    # We need real source media for the video stage path resolution,
    # but samples + generate don't read it. Skip the source symlink
    # and call _run_single_stage directly.
    wsp._run_single_stage(
        manifest2, "samples", ctx2, workspace_root=workspace_root,
    )
    assert "speaker_profiles" in ctx2, (
        f"samples stage's return value was not merged into the context. "
        f"context keys: {sorted(ctx2._data.keys())!r}"
    )
    assert ctx2["speaker_profiles"] == {
        "speaker_01": {"profiles": {"primary": "REF"}}
    }
    wsp._run_single_stage(
        manifest2, "generate", ctx2, workspace_root=workspace_root,
    )
    assert captured["saw_in_generate"].get("speaker_profiles") == {
        "speaker_01": {"profiles": {"primary": "REF"}}
    }, (
        f"generate stage did not see speaker_profiles. "
        f"generate context keys: {sorted(captured['saw_in_generate'].keys())!r}"
    )
