"""End-to-end integration test for :meth:`WorkspacePipeline.prepare`.

The real stage classes (pyannote, whisper, omnivoice) require GPU + heavy
models and cannot run in CI. This test replaces every entry in
:data:`src.workspace.pipeline.STAGE_CLASSES` with an in-process stub that
records its invocation and materialises minimal output files, so the
pipeline's bookkeeping (workspace layout, manifest, metadata, atomic
promote) is exercised end-to-end.

The five class attributes the pipeline reads (``inputs``, ``outputs``,
``editable_outputs``, ``derived_outputs``, ``config_fields``) are declared
on the stub so the manifest records the right lineage for the invalidation
DAG to consume in the edit-scenario tests.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest
import soundfile as sf

# Make `src` importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Stage I/O tables (workspace-relative paths).
#
# These mirror the canonical manifest constructed in
# ``tests/test_workspace_dag.py`` so the invalidation DAG in
# ``generate()`` sees the same producer/consumer chain that the
# edit-scenario tests verify. The pipeline reads them via
# ``STAGE_CLASSES[name].inputs`` / ``.outputs`` when building ArtifactRefs.
# ---------------------------------------------------------------------------

STAGE_INPUTS: dict[str, List[str]] = {
    "extract": [],
    "separate": ["media/original_audio.wav"],
    "diarize": ["media/speech.wav"],
    "samples": ["media/speech.wav", "diarization/segments.json"],
    "transcribe": ["media/speech.wav", "diarization/segments.json"],
    "translate": [
        "transcription/transcript.json",
        "translation/glossary.json",
    ],
    "generate": [
        "translation/translated_transcript.json",
        "speakers/speaker_01/primary.wav",
        "speakers/speaker_01/primary.txt",
    ],
    "align": ["generated_segments/manifest.json"],
    "reconstruct": ["aligned_manifest.json", "media/background.wav"],
    "mix": ["output/reconstructed_speech.wav", "media/background.wav"],
    "video": ["output/final_audio.wav"],
}

STAGE_OUTPUTS: dict[str, List[str]] = {
    "extract": ["media/original_audio.wav", "media_info.json"],
    "separate": ["media/speech.wav", "media/background.wav"],
    "diarize": [
        "diarization/segments.json",
        "diarization/embeddings.npz",
        "diarization/embeddings.meta.json",
        "diarization/metadata.json",
    ],
    "samples": [
        "speakers/speaker_01/primary.wav",
        "speakers/speaker_01/primary.txt",
        "speakers/speaker_01/embedding.npy",
        "speakers/speaker_01/metadata.json",
        "speakers/speaker_01/candidates/candidate_01.wav",
        "speakers/speaker_01/candidates/candidate_01.txt",
        "speakers/speaker_01/candidates/candidate_01.score.json",
    ],
    "transcribe": [
        "transcription/transcript.json",
        "transcription/word_level.json",
    ],
    "translate": [
        "translation/translated_transcript.json",
        "translation/glossary.json",
    ],
    "generate": ["generated_segments/manifest.json"],
    "align": ["aligned_manifest.json"],
    "reconstruct": ["output/reconstructed_speech.wav"],
    "mix": ["output/final_audio.wav"],
    "video": ["output/final_video.mp4"],
}

STAGE_EDITABLE: dict[str, List[str]] = {
    "extract": [],
    "separate": [],
    "diarize": [],
    "samples": [
        "speakers/speaker_01/primary.wav",
        "speakers/speaker_01/primary.txt",
    ],
    "transcribe": ["transcription/transcript.json"],
    "translate": [
        "translation/translated_transcript.json",
        "translation/glossary.json",
    ],
    "generate": [],
    "align": [],
    "reconstruct": [],
    "mix": [],
    "video": [],
}

STAGE_DERIVED: dict[str, List[str]] = {
    "extract": [],
    "separate": [],
    "diarize": ["diarization/embeddings.npz"],
    "samples": [],
    "transcribe": [],
    "translate": [],
    "generate": ["generated_segments/manifest.json"],
    "align": ["aligned_manifest.json"],
    "reconstruct": ["output/reconstructed_speech.wav"],
    "mix": ["output/final_audio.wav"],
    "video": ["output/final_video.mp4"],
}

STAGE_CONFIG: dict[str, List[str]] = {
    "extract": ["sample_rate"],
    "separate": [],
    "diarize": [],
    "samples": [],
    "transcribe": [],
    "translate": ["source_language", "target_language"],
    "generate": ["tts_backend"],
    "align": [],
    "reconstruct": [],
    "mix": [],
    "video": [],
}


# ---------------------------------------------------------------------------
# Stub factory
# ---------------------------------------------------------------------------


def make_stub(
    stage_name: str,
    output_paths: List[str],
    record: List[str],
    *,
    inputs: List[str],
    editable_outputs: List[str],
    derived_outputs: List[str],
    config_fields: List[str],
):
    """Build a self-contained stub class that mimics the real stage API.

    The stub declares the five class attributes the pipeline reads from
    :data:`STAGE_CLASSES` and a :meth:`run` method that records its name
    in ``record`` and materialises minimal output files inside
    ``self.workdir``. The only third-party imports are ``numpy`` and
    ``soundfile``; the constructor swallows every keyword the orchestrator
    passes (whisper model, hf_token, output_dir, etc.) so a single stub
    shape works for all 11 stages.

    We build the class with :func:`type` rather than a ``class`` block
    because Python's class body does not see the enclosing function's
    locals for names it is about to assign to (e.g. ``inputs = inputs``
    raises ``NameError``).
    """

    class _StubBase:
        def __init__(self, workdir, *args, subdir=None, **kwargs):
            # The orchestrator overwrites ``self.workdir`` (and, for
            # reconstruct/mix/video, ``self.output_dir``) right after
            # construction, so we just store whatever the constructor was
            # given and let the redirection take effect.
            self.workdir = Path(workdir)
            self._subdir = subdir
            if subdir:
                self.workdir = self.workdir / subdir
            # Record any kwargs the orchestrator passed (e.g. ``model_id``,
            # ``sample_rate``) so the manifest's config block can be
            # populated by ``getattr(stage, k)`` in the pipeline.
            for k, v in kwargs.items():
                setattr(self, k, v)

        def output_paths(self):
            return []

        def run(self, *args, **kwargs):
            record.append(stage_name)
            self.workdir.mkdir(parents=True, exist_ok=True)
            for p in output_paths:
                # ``p`` is a *workspace-relative* path (e.g.
                # ``speakers/speaker_01/primary.wav``).  The orchestrator
                # has already redirected ``self.workdir`` to either
                # ``staging/<subdir>`` (e.g. for ``extract`` ->
                # ``staging/media``) or ``staging`` (for stages without
                # a subdir, e.g. ``samples`` whose output already
                # contains the full ``speakers/`` prefix).  Strip the
                # subdir prefix when present so the file lands in the
                # right place for ``promote()`` to materialise the
                # canonical workspace layout.
                rel = p
                sub = self._subdir
                if sub and (rel == sub or rel.startswith(sub + "/")):
                    rel = rel[len(sub) + 1:] if rel != sub else ""
                target = (self.workdir / rel) if rel else self.workdir
                target.parent.mkdir(parents=True, exist_ok=True)
                if p.endswith(".wav"):
                    sf.write(
                        str(target),
                        np.zeros(8000, dtype="float32"),
                        16000,
                        subtype="PCM_16",
                    )
                elif p.endswith(".npz") or p.endswith(".npy"):
                    np.save(str(target), np.zeros(4, dtype="float32"))
                elif p.endswith(".json"):
                    target.write_text(json.dumps({"speakers": [], "segments": []}))
                else:
                    target.write_bytes(b"x")
            return {}

    return type(
        "Stub",
        (_StubBase,),
        {
            "name": stage_name,
            "inputs": inputs,
            "outputs": output_paths,
            "editable_outputs": editable_outputs,
            "derived_outputs": derived_outputs,
            "config_fields": config_fields,
        },
    )


def install_stubs(monkeypatch, record):
    """Replace every entry in :data:`STAGE_CLASSES` with a fresh stub.

    The pipeline's :func:`src.workspace.pipeline._build_stage` is hard-wired
    to the real stage class names (it imports ``ExtractStage`` etc. and
    picks the constructor with an if/elif chain). We monkey-patch that
    function too, so the stub in :data:`STAGE_CLASSES` is the class that
    actually gets instantiated.
    """
    from src.workspace import pipeline as wsp_mod

    def patched_build_stage(
        name,
        workspace_root,
        *,
        no_pyannote,
        hf_token,
        whisper_model,
        source_language,
        target_language,
        min_speakers,
        max_speakers,
        glossary_path,
        target_lufs,
        tts_backend="omnivoice",
        stage_overrides=None,
    ):
        cls = wsp_mod.STAGE_CLASSES[name]
        subdir = wsp_mod.stage_subdir(name)
        # The stubs accept ``(workdir, *args, subdir=None, **kwargs)``,
        # so we can pass the orchestrator's kwargs straight through. We
        # do not need to forward ``output_dir`` — the pipeline sets
        # ``stage.output_dir`` on the instance after construction.
        return cls(
            workspace_root,
            no_pyannote=no_pyannote,
            hf_token=hf_token,
            whisper_model=whisper_model,
            source_language=source_language,
            target_language=target_language,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            glossary_path=glossary_path,
            target_lufs=target_lufs,
            tts_backend=tts_backend,
            subdir=subdir,
        )

    monkeypatch.setattr(wsp_mod, "_build_stage", patched_build_stage)

    # Build a new STAGE_CLASSES dict so the pipeline's module-globals
    # lookup picks up our stubs (mutating the original dict in place
    # also works in principle, but replacing the binding is more
    # robust against any caching the runtime might do).
    new_classes = {}
    for stage_name, out_paths in STAGE_OUTPUTS.items():
        stub = make_stub(
            stage_name,
            out_paths,
            record,
            inputs=STAGE_INPUTS[stage_name],
            editable_outputs=STAGE_EDITABLE[stage_name],
            derived_outputs=STAGE_DERIVED[stage_name],
            config_fields=STAGE_CONFIG[stage_name],
        )
        new_classes[stage_name] = stub
    monkeypatch.setattr(wsp_mod, "STAGE_CLASSES", new_classes)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fixture_audio():
    """Path to the 30 s mono 16 kHz test fixture (Task 1 deliverable)."""
    p = Path(__file__).resolve().parent / "fixtures" / "short_sample.wav"
    if not p.exists():
        import subprocess

        subprocess.check_call(
            [
                "python",
                str(
                    Path(__file__).resolve().parent
                    / "fixtures"
                    / "generate_short_sample.py"
                ),
            ]
        )
    return str(p)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_prepare_creates_workspace_with_all_artifacts(
    monkeypatch, tmp_path, fixture_audio
):
    """``prepare()`` runs all 11 stages and materialises the canonical layout."""
    from src.workspace import pipeline as wsp_mod
    from src.workspace.pipeline import WorkspacePipeline

    monkeypatch.setenv("AI_DUBBING_WORKSPACES_ROOT", str(tmp_path))
    record: List[str] = []
    install_stubs(monkeypatch, record)

    wsp = WorkspacePipeline(
        input_path=fixture_audio,
        source_language="en",
        target_language="es",
    )
    wid, root = wsp.prepare()

    # Top-level metadata + manifest files exist.
    assert (root / "metadata.json").exists()
    assert (root / "manifest.json").exists()

    # A handful of representative artefacts from across the pipeline.
    assert (root / "media" / "original_audio.wav").exists()
    assert (root / "translation" / "translated_transcript.json").exists()

    # The manifest records the six ``prepare`` stages (extract..translate)
    # as ``done`` — i.e. every stub's ``run`` was invoked, the staging dir
    # was promoted, and the stage record was saved.  The five
    # ``generate`` stages (generate..video) are present in the manifest
    # with status ``pending`` because the manifest is pre-populated with
    # all 11 stages' editable/derived paths; they get their ``run`` calls
    # when the user invokes ``dub.sh generate <wid>``.
    m = json.loads((root / "manifest.json").read_text())
    assert len(m["stages"]) == 11
    prepare_stages = {"extract", "separate", "diarize", "samples", "transcribe", "translate"}
    for s in m["stages"].values():
        name = s["name"]
        if name in prepare_stages:
            assert s["status"] == "done", f"prepare stage {name!r} not done: {s}"
        else:
            assert s["status"] == "pending", f"generate stage {name!r} should be pending: {s}"

    # And the stub recorder agrees.
    assert record == [
        "extract", "separate", "diarize", "samples", "transcribe", "translate"
    ]
