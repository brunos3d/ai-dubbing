"""Integration tests for the spec §10 invalidation DAG end-to-end.

Like ``test_workspace_e2e.py`` this module monkey-patches
:data:`src.workspace.pipeline.STAGE_CLASSES` with in-process stubs so the
``WorkspacePipeline`` can be exercised without GPU or heavy models.  Where
``test_workspace_e2e`` covers the *layout* of the workspace after
``prepare()``, this module covers the *behaviour* of ``generate()`` after a
user has edited one of the editable artefacts — exactly the workflow the
spec §10 table is built around.

Three scenarios are tested, mirroring three of the eight rows in
``tests/test_workspace_dag.py``'s release-blocker table:

* editing ``translation/translated_transcript.json`` (editable) re-runs only
  ``generate..video``;
* editing ``speakers/speaker_01/primary.wav`` (editable) re-runs only
  ``generate..video``;
* editing ``diarization/segments.json`` (non-editable) re-runs the full
  downstream cascade from ``diarize`` onwards.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Stage I/O tables (mirror the canonical manifest in
# ``tests/test_workspace_dag.py``).
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
# Stub factory (mirrors test_workspace_e2e.py)
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
    class _StubBase:
        def __init__(self, workdir, *args, subdir=None, **kwargs):
            self.workdir = Path(workdir)
            self._subdir = subdir
            if subdir:
                self.workdir = self.workdir / subdir
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
                    target.write_text(
                        json.dumps({"speakers": [], "segments": []})
                    )
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
    ):
        cls = wsp_mod.STAGE_CLASSES[name]
        subdir = wsp_mod.stage_subdir(name)
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


@pytest.fixture
def prepared_workspace(monkeypatch, tmp_path, fixture_audio):
    """Run ``prepare()`` with stubs in place; return ``(record, wid, root)``."""
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
    return record, wid, root


# ---------------------------------------------------------------------------
# Tests (mirror §10 of the spec)
# ---------------------------------------------------------------------------


def test_edit_translation_triggers_only_post_translate(
    monkeypatch, tmp_path, prepared_workspace
):
    """Row 1: editing ``translated_transcript.json`` re-runs only
    ``generate..video``.
    """
    record_pre, wid, root = prepared_workspace
    # Sanity: the prepare pass ran the 6 expected stages.
    assert record_pre == [
        "extract", "separate", "diarize", "samples", "transcribe", "translate"
    ]

    # Mutate the editable translation file.
    target = root / "translation" / "translated_transcript.json"
    target.write_text(json.dumps({"segments": [{"text": "EDITED"}]}))

    # Re-run generate with a fresh record.
    from src.workspace.pipeline import WorkspacePipeline

    record: List[str] = []
    install_stubs(monkeypatch, record)
    wsp = WorkspacePipeline(
        input_path=str(tmp_path / "dummy.wav"),
        source_language="en",
        target_language="es",
    )
    wsp.generate(wid)

    assert record == ["generate", "align", "reconstruct", "mix", "video"]


def test_edit_speaker_primary_triggers_only_post_translate(
    monkeypatch, tmp_path, prepared_workspace
):
    """Row 2: editing ``speakers/speaker_01/primary.wav`` re-runs only
    ``generate..video``.
    """
    _record_pre, wid, root = prepared_workspace

    # Mutate the editable speaker primary.
    target = root / "speakers" / "speaker_01" / "primary.wav"
    sf.write(str(target), np.zeros(16000, dtype="float32"), 16000, subtype="PCM_16")

    from src.workspace.pipeline import WorkspacePipeline

    record: List[str] = []
    install_stubs(monkeypatch, record)
    wsp = WorkspacePipeline(
        input_path=str(tmp_path / "dummy.wav"),
        source_language="en",
        target_language="es",
    )
    wsp.generate(wid)

    assert record == ["generate", "align", "reconstruct", "mix", "video"]


def test_edit_diarization_segments_triggers_full_chain(
    monkeypatch, tmp_path, prepared_workspace
):
    """Row 5: editing ``diarization/segments.json`` (non-editable) re-runs
    the producer (``diarize``) and every downstream consumer down to
    ``video``.
    """
    _record_pre, wid, root = prepared_workspace

    target = root / "diarization" / "segments.json"
    target.write_text(
        json.dumps({"speakers": ["X"], "segments": [{"start": 0.0, "end": 1.0, "speaker": "X"}]})
    )

    from src.workspace.pipeline import WorkspacePipeline

    record: List[str] = []
    install_stubs(monkeypatch, record)
    wsp = WorkspacePipeline(
        input_path=str(tmp_path / "dummy.wav"),
        source_language="en",
        target_language="es",
    )
    wsp.generate(wid)

    # Per spec §10 step 2, the producer (``diarize``) is also marked stale
    # when a non-editable input hash mismatches.  The plan's verified-
    # scenarios table summarises the consumer chain; the algorithm is
    # authoritative.
    assert record == [
        "diarize", "samples", "transcribe", "translate",
        "generate", "align", "reconstruct", "mix", "video",
    ]
