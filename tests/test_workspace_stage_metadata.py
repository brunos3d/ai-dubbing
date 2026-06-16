"""Tests for the workspace metadata class attributes on all 11 stages.

Each stage class must expose five class attributes so the WorkspacePipeline
can build its manifest, invalidate stale outputs, and detect editable files
without inspecting stage bodies:
    - inputs         (list[str])   relative paths the stage reads
    - outputs        (list[str])   relative paths the stage writes
    - editable_outputs (list[str]) relative paths a user may edit
    - derived_outputs  (list[str]) relative paths derived from earlier outputs
    - config_fields  (list[str])   names of __init__ kwargs that affect output

And every ``__init__`` must accept an optional ``subdir`` keyword that, when
truthy, reassigns ``self.workdir`` to ``self.workdir / subdir`` so the
WorkspacePipeline can build a nested workspace layout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stages.align import AlignStage  # noqa: E402
from src.stages.diarize import DiarizeStage  # noqa: E402
from src.stages.extract import ExtractStage  # noqa: E402
from src.stages.generate import GenerateStage  # noqa: E402
from src.stages.mix import MixStage  # noqa: E402
from src.stages.reconstruct import ReconstructStage  # noqa: E402
from src.stages.samples import SampleStage  # noqa: E402
from src.stages.separate import SeparateStage  # noqa: E402
from src.stages.transcribe import TranscribeStage  # noqa: E402
from src.stages.translate import TranslateStage  # noqa: E402
from src.stages.video import VideoStage  # noqa: E402


EXPECTED: dict[type, dict] = {
    ExtractStage: {
        "inputs": [],
        "outputs": ["media/original_audio.wav", "media_info.json"],
        "editable_outputs": [],
        "derived_outputs": [],
        "config_fields": ["sample_rate"],
    },
    SeparateStage: {
        "inputs": ["media/original_audio.wav"],
        "outputs": ["media/speech.wav", "media/background.wav"],
        "editable_outputs": [],
        "derived_outputs": [],
        "config_fields": ["model", "device", "out_dir_name"],
    },
    DiarizeStage: {
        "inputs": ["media/speech.wav"],
        "outputs": [
            "diarization/segments.json",
            "diarization/embeddings.npz",
            "diarization/embeddings.meta.json",
            "diarization/metadata.json",
        ],
        "editable_outputs": [],
        "derived_outputs": ["diarization/embeddings.npz"],
        "config_fields": ["model_id", "min_speakers", "max_speakers", "no_pyannote"],
    },
    SampleStage: {
        "inputs": ["media/speech.wav", "diarization/segments.json"],
        "outputs": [
            "speakers/speaker_01/primary.wav",
            "speakers/speaker_01/primary.txt",
            "speakers/speaker_01/embedding.npy",
            "speakers/speaker_01/metadata.json",
            "speakers/speaker_01/candidates/candidate_01.wav",
            "speakers/speaker_01/candidates/candidate_01.txt",
            "speakers/speaker_01/candidates/candidate_01.score.json",
        ],
        "editable_outputs": [
            "speakers/speaker_01/primary.wav",
            "speakers/speaker_01/primary.txt",
        ],
        "derived_outputs": [],
        "config_fields": ["target_seconds", "max_seconds"],
    },
    TranscribeStage: {
        "inputs": ["media/speech.wav", "diarization/segments.json"],
        "outputs": [
            "transcription/transcript.json",
            "transcription/word_level.json",
        ],
        "editable_outputs": ["transcription/transcript.json"],
        "derived_outputs": [],
        "config_fields": ["model_size", "source_language", "device"],
    },
    TranslateStage: {
        "inputs": ["transcription/transcript.json"],
        "outputs": [
            "translation/translated_transcript.json",
            "translation/glossary.json",
            "translation/timing_report.json",
        ],
        "editable_outputs": [
            "translation/translated_transcript.json",
            "translation/glossary.json",
        ],
        "derived_outputs": ["translation/timing_report.json"],
        "config_fields": ["source_language", "target_language", "backend_name"],
    },
    GenerateStage: {
        "inputs": ["translation/translated_transcript.json"],
        "outputs": ["generated_segments/manifest.json"],
        "editable_outputs": [],
        "derived_outputs": ["generated_segments/manifest.json"],
        "config_fields": ["model_id", "target_language", "use_clone_prompt"],
    },
    AlignStage: {
        "inputs": ["generated_segments/manifest.json"],
        "outputs": ["aligned_manifest.json"],
        "editable_outputs": [],
        "derived_outputs": ["aligned_manifest.json"],
        "config_fields": [
            "target_language",
            "tolerance",
            "regenerate_with_duration",
        ],
    },
    ReconstructStage: {
        "inputs": [
            "aligned_manifest.json",
            "media/background.wav",
        ],
        "outputs": ["output/reconstructed_speech.wav"],
        "editable_outputs": [],
        "derived_outputs": ["output/reconstructed_speech.wav"],
        "config_fields": ["target_sr"],
    },
    MixStage: {
        "inputs": [
            "output/reconstructed_speech.wav",
            "media/background.wav",
        ],
        "outputs": ["output/final_audio.wav"],
        "editable_outputs": [],
        "derived_outputs": ["output/final_audio.wav"],
        "config_fields": ["target_lufs", "speech_db", "background_db"],
    },
    VideoStage: {
        "inputs": ["output/final_audio.wav"],
        "outputs": ["output/final_video.mp4"],
        "editable_outputs": [],
        "derived_outputs": ["output/final_video.mp4"],
        "config_fields": [],
    },
}


_ATTRS = ("inputs", "outputs", "editable_outputs", "derived_outputs", "config_fields")


@pytest.mark.parametrize("cls,expected", list(EXPECTED.items()))
def test_stage_metadata_matches_table(cls, expected):
    for attr in _ATTRS:
        got = getattr(cls, attr)
        assert got == expected[attr], (
            f"{cls.__name__}.{attr} mismatch:\n"
            f"  expected: {sorted(expected[attr]) if isinstance(expected[attr], list) else expected[attr]!r}\n"
            f"  got:      {sorted(got) if isinstance(got, list) else got!r}"
        )


def test_editable_outputs_listed_in_editable_paths():
    """The union of all editable_outputs must cover every editable path.

    Users can hand-edit transcripts, translations, the glossary, and the
    primary speaker sample. Every one of these must appear in at least one
    stage's ``editable_outputs`` so the invalidation DAG can pick up the
    change.
    """
    must_be_editable = {
        "transcription/transcript.json",
        "translation/translated_transcript.json",
        "translation/glossary.json",
        "speakers/speaker_01/primary.wav",
        "speakers/speaker_01/primary.txt",
    }
    union: set[str] = set()
    for cls in EXPECTED:
        for p in cls.editable_outputs:
            union.add(p)
    missing = must_be_editable - union
    assert not missing, f"editable paths missing from any stage's editable_outputs: {sorted(missing)}"



