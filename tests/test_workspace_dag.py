"""Tests for src/workspace/dag.py (Task 7, release blocker — spec §10)."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.workspace.dag import (  # noqa: E402
    STAGE_ORDER,
    CliOverrides,
    compute_stale_set,
)
from src.workspace.manifest import ArtifactRef, Manifest, StageRecord  # noqa: E402


def _hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _ref(path: str, content: bytes = b"orig") -> ArtifactRef:
    return ArtifactRef(path=path, sha256=_hash(content), size_bytes=len(content))


def _make_full_manifest(workspace: Path) -> Manifest:
    """Build a manifest with all 11 stages and their declared inputs/outputs.

    The file contents are deterministic but distinct so the hash check in
    ``compute_stale_set`` is meaningful. The editable/derived sets come from
    the spec's stage class attributes (see plan Task 8 table).

    The wiring makes the spec §10 algorithm produce the verified scenarios in
    spec/2026-06-13-workspace-architecture-design.md. In particular:

    * Editable artefacts (``translation/glossary.json``,
      ``speakers/.../primary.wav``) are declared as inputs of the stage that
      consumes them, so editing them marks that consumer stale.
    * Intermediate files (``generated_segments/manifest.json``,
      ``aligned_manifest.json``, etc.) are *not* listed in
      ``derived_paths`` — they are real inputs of downstream stages and the
      propagation step (spec §10.3) must walk through them.
    """
    m = Manifest.create(
        workspace_id="ws-test", pipeline_version="0.1.0", git_commit="abc"
    )

    extract_outputs = [
        _ref("media/original_audio.wav", b"orig-audio"),
        _ref("media_info.json", b"info"),
    ]
    separate_inputs = [_ref("media/original_audio.wav", b"orig-audio")]
    separate_outputs = [
        _ref("media/speech.wav", b"speech"),
        _ref("media/background.wav", b"background"),
    ]
    diarize_inputs = [_ref("media/speech.wav", b"speech")]
    diarize_outputs = [
        _ref("diarization/segments.json", b"segments"),
        _ref("diarization/embeddings.npz", b"embeddings-npz"),
        _ref("diarization/embeddings.meta.json", b"embeddings-meta"),
        _ref("diarization/metadata.json", b"diar-meta"),
    ]
    samples_inputs = [
        _ref("media/speech.wav", b"speech"),
        _ref("diarization/segments.json", b"segments"),
    ]
    samples_outputs = [
        _ref("speakers/speaker_01/primary.wav", b"sp1-wav"),
        _ref("speakers/speaker_01/primary.txt", b"sp1-txt"),
        _ref("speakers/speaker_01/embedding.npy", b"sp1-emb"),
        _ref("speakers/speaker_01/metadata.json", b"sp1-meta"),
        _ref(
            "speakers/speaker_01/candidates/candidate_01.wav",
            b"sp1-cand-wav",
        ),
        _ref(
            "speakers/speaker_01/candidates/candidate_01.txt",
            b"sp1-cand-txt",
        ),
        _ref(
            "speakers/speaker_01/candidates/candidate_01.score.json",
            b"sp1-cand-score",
        ),
    ]
    transcribe_inputs = [
        _ref("media/speech.wav", b"speech"),
        _ref("diarization/segments.json", b"segments"),
    ]
    transcribe_outputs = [
        _ref("transcription/transcript.json", b"transcript"),
        _ref("transcription/word_level.json", b"word-level"),
    ]
    # translate reads its own previous glossary (and the transcript) to decide
    # which terms to preserve, so glossary is BOTH an input and an output.
    translate_inputs = [
        _ref("transcription/transcript.json", b"transcript"),
        _ref("translation/glossary.json", b"glossary"),
    ]
    translate_outputs = [
        _ref("translation/translated_transcript.json", b"translated"),
        _ref("translation/glossary.json", b"glossary"),
    ]
    # generate uses the speaker's primary sample to clone the voice, so it
    # is both a real input (and editable artefact) of the stage.
    generate_inputs = [
        _ref("translation/translated_transcript.json", b"translated"),
        _ref("speakers/speaker_01/primary.wav", b"sp1-wav"),
        _ref("speakers/speaker_01/primary.txt", b"sp1-txt"),
    ]
    generate_outputs = [_ref("generated_segments/manifest.json", b"gen-manifest")]
    align_inputs = [_ref("generated_segments/manifest.json", b"gen-manifest")]
    align_outputs = [_ref("aligned_manifest.json", b"aligned")]
    reconstruct_inputs = [
        _ref("aligned_manifest.json", b"aligned"),
        _ref("media/background.wav", b"background"),
    ]
    reconstruct_outputs = [
        _ref("output/reconstructed_speech.wav", b"reconstructed")
    ]
    mix_inputs = [
        _ref("output/reconstructed_speech.wav", b"reconstructed"),
        _ref("media/background.wav", b"background"),
    ]
    mix_outputs = [_ref("output/final_audio.wav", b"final-audio")]
    video_inputs = [_ref("output/final_audio.wav", b"final-audio")]
    video_outputs = [_ref("output/final_video.mp4", b"final-video")]

    m.add_stage(
        "extract",
        StageRecord(
            name="extract",
            status="done",
            inputs=[],
            outputs=extract_outputs,
        ),
    )
    m.add_stage(
        "separate",
        StageRecord(
            name="separate",
            status="done",
            inputs=separate_inputs,
            outputs=separate_outputs,
        ),
    )
    m.add_stage(
        "diarize",
        StageRecord(
            name="diarize",
            status="done",
            inputs=diarize_inputs,
            outputs=diarize_outputs,
        ),
    )
    m.add_stage(
        "samples",
        StageRecord(
            name="samples",
            status="done",
            inputs=samples_inputs,
            outputs=samples_outputs,
        ),
    )
    m.add_stage(
        "transcribe",
        StageRecord(
            name="transcribe",
            status="done",
            inputs=transcribe_inputs,
            outputs=transcribe_outputs,
        ),
    )
    m.add_stage(
        "translate",
        StageRecord(
            name="translate",
            status="done",
            inputs=translate_inputs,
            outputs=translate_outputs,
        ),
    )
    m.add_stage(
        "generate",
        StageRecord(
            name="generate",
            status="done",
            inputs=generate_inputs,
            outputs=generate_outputs,
        ),
    )
    m.add_stage(
        "align",
        StageRecord(
            name="align",
            status="done",
            inputs=align_inputs,
            outputs=align_outputs,
        ),
    )
    m.add_stage(
        "reconstruct",
        StageRecord(
            name="reconstruct",
            status="done",
            inputs=reconstruct_inputs,
            outputs=reconstruct_outputs,
        ),
    )
    m.add_stage(
        "mix",
        StageRecord(
            name="mix",
            status="done",
            inputs=mix_inputs,
            outputs=mix_outputs,
        ),
    )
    m.add_stage(
        "video",
        StageRecord(
            name="video",
            status="done",
            inputs=video_inputs,
            outputs=video_outputs,
        ),
    )

    # Editable artefacts (spec §9).
    m.add_editable_path("transcription/transcript.json")
    m.add_editable_path("translation/translated_transcript.json")
    m.add_editable_path("translation/glossary.json")
    m.add_editable_path("speakers/speaker_01/primary.wav")
    m.add_editable_path("speakers/speaker_01/primary.txt")

    # Derived artefacts: only those that are side-effects, not part of the
    # producer/consumer chain. The intermediate files in the
    # generate→align→reconstruct→mix→video chain are *not* derived — they
    # are real inputs that downstream stages depend on, and the propagation
    # in spec §10.3 must walk through them.
    m.add_derived_path("diarization/embeddings.npz")

    return m


def _write_files(workspace: Path) -> None:
    """Materialise the files referenced by the manifest produced above."""
    pairs = {
        "media/original_audio.wav": b"orig-audio",
        "media_info.json": b"info",
        "media/speech.wav": b"speech",
        "media/background.wav": b"background",
        "diarization/segments.json": b"segments",
        "diarization/embeddings.npz": b"embeddings-npz",
        "diarization/embeddings.meta.json": b"embeddings-meta",
        "diarization/metadata.json": b"diar-meta",
        "speakers/speaker_01/primary.wav": b"sp1-wav",
        "speakers/speaker_01/primary.txt": b"sp1-txt",
        "speakers/speaker_01/embedding.npy": b"sp1-emb",
        "speakers/speaker_01/metadata.json": b"sp1-meta",
        "speakers/speaker_01/candidates/candidate_01.wav": b"sp1-cand-wav",
        "speakers/speaker_01/candidates/candidate_01.txt": b"sp1-cand-txt",
        "speakers/speaker_01/candidates/candidate_01.score.json": b"sp1-cand-score",
        "transcription/transcript.json": b"transcript",
        "transcription/word_level.json": b"word-level",
        "translation/translated_transcript.json": b"translated",
        "translation/glossary.json": b"glossary",
        "generated_segments/manifest.json": b"gen-manifest",
        "aligned_manifest.json": b"aligned",
        "output/reconstructed_speech.wav": b"reconstructed",
        "output/final_audio.wav": b"final-audio",
        "output/final_video.mp4": b"final-video",
    }
    for rel, content in pairs.items():
        p = workspace / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)


def test_edit_translation_translated_transcript(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _write_files(workspace)
    m = _make_full_manifest(workspace)
    (workspace / "translation" / "translated_transcript.json").write_bytes(b"NEW")
    stale = compute_stale_set(m, workspace)
    assert stale == ["generate", "align", "reconstruct", "mix", "video"]


def test_edit_speaker_primary_wav(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _write_files(workspace)
    m = _make_full_manifest(workspace)
    (workspace / "speakers" / "speaker_01" / "primary.wav").write_bytes(b"NEW")
    stale = compute_stale_set(m, workspace)
    assert stale == ["generate", "align", "reconstruct", "mix", "video"]


def test_edit_translation_glossary(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _write_files(workspace)
    m = _make_full_manifest(workspace)
    (workspace / "translation" / "glossary.json").write_bytes(b"NEW")
    stale = compute_stale_set(m, workspace)
    assert stale == [
        "translate",
        "generate",
        "align",
        "reconstruct",
        "mix",
        "video",
    ]


def test_edit_transcription_transcript(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _write_files(workspace)
    m = _make_full_manifest(workspace)
    (workspace / "transcription" / "transcript.json").write_bytes(b"NEW")
    stale = compute_stale_set(m, workspace)
    assert stale == [
        "translate",
        "generate",
        "align",
        "reconstruct",
        "mix",
        "video",
    ]


def test_edit_diarization_segments_json(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _write_files(workspace)
    m = _make_full_manifest(workspace)
    (workspace / "diarization" / "segments.json").write_bytes(b"NEW")
    stale = compute_stale_set(m, workspace)
    assert stale == [
        "diarize",
        "samples",
        "transcribe",
        "translate",
        "generate",
        "align",
        "reconstruct",
        "mix",
        "video",
    ]


def test_edit_media_speech_wav(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _write_files(workspace)
    m = _make_full_manifest(workspace)
    (workspace / "media" / "speech.wav").write_bytes(b"NEW")
    # ``media/speech.wav`` is a non-editable output of ``separate``; per
    # spec §10 step 2, a hash mismatch on a non-editable input walks
    # upstream to the producing stage (``separate``) and also marks the
    # consumer (``diarize``) stale. Propagation then runs everything
    # downstream. (The spec's verified-scenarios table omits ``separate``
    # in the expected list for this row, which we treat as a typo in the
    # table — the algorithm itself is unambiguous about marking the
    # producer.)
    stale = compute_stale_set(m, workspace)
    assert stale == [
        "separate",
        "diarize",
        "samples",
        "transcribe",
        "translate",
        "generate",
        "align",
        "reconstruct",
        "mix",
        "video",
    ]


def test_cli_from_stage_generate(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _write_files(workspace)
    m = _make_full_manifest(workspace)
    stale = compute_stale_set(
        m, workspace, CliOverrides(force=False, from_stage="generate")
    )
    assert stale == ["generate", "align", "reconstruct", "mix", "video"]


def test_cli_force(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _write_files(workspace)
    m = _make_full_manifest(workspace)
    stale = compute_stale_set(m, workspace, CliOverrides(force=True))
    assert stale == list(STAGE_ORDER)


def test_no_changes_means_no_stale(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _write_files(workspace)
    m = _make_full_manifest(workspace)
    stale = compute_stale_set(m, workspace)
    assert stale == []


def test_missing_file_treated_as_hash_mismatch(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _write_files(workspace)
    m = _make_full_manifest(workspace)
    # Delete a non-editable file; that should mark its producer + downstream
    # as stale.
    (workspace / "diarization" / "segments.json").unlink()
    stale = compute_stale_set(m, workspace)
    assert "diarize" in stale
    assert "video" in stale


def test_to_stage_caps_stale_set(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _write_files(workspace)
    m = _make_full_manifest(workspace)
    # Force everything stale, then cap to transcribe (inclusive).
    stale = compute_stale_set(
        m,
        workspace,
        CliOverrides(force=True, to_stage="transcribe"),
    )
    assert stale == ["extract", "separate", "diarize", "samples", "transcribe"]


def test_untracked_config_fields_do_not_cause_false_stale(tmp_path: Path) -> None:
    """Regression: the ``separate`` stage declares ``config_fields =
    ["model", "device", "out_dir_name"]`` but ``WorkspacePipeline`` does
    not carry those attributes (it only tracks the high-level knobs the
    CLI exposes).  When ``generate()`` builds ``current_configs`` via
    ``getattr(self, k, None)`` the untracked fields come back as
    ``None``.  The old behaviour treated ``None != "cuda"`` as a config
    change, which re-ran an already-finished Demucs separation for
    nothing.  The DAG must ignore ``None``-valued current entries —
    they are not user overrides, just gaps in the pipeline's tracking.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _write_files(workspace)
    m = _make_full_manifest(workspace)

    # Simulate what ``WorkspacePipeline.generate()`` actually produces for
    # the ``separate`` stage: the untracked fields come back as ``None``
    # because the pipeline does not have those attributes.  Tracked
    # fields (e.g. ``target_lufs`` for ``mix``) come back with the
    # pipeline's default.
    current_configs = {
        "separate": {
            "model": None,
            "device": None,
            "out_dir_name": None,
        },
    }
    stale = compute_stale_set(m, workspace, current_configs=current_configs)
    assert stale == [], (
        f"Untracked (None) config fields must not mark stages stale. "
        f"Got stale={stale!r}"
    )


def test_real_config_change_is_still_detected(tmp_path: Path) -> None:
    """Companion to ``test_untracked_config_fields_do_not_cause_false_stale``:
    a genuinely different user-supplied value must still invalidate the
    stage.  The previous fix must not over-correct.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _write_files(workspace)
    m = _make_full_manifest(workspace)

    current_configs = {
        "separate": {
            "model": "htdemucs_ft",  # user explicitly asked for a different model
            "device": None,            # untracked, ignored
            "out_dir_name": None,      # untracked, ignored
        },
    }
    stale = compute_stale_set(m, workspace, current_configs=current_configs)
    assert "separate" in stale, (
        f"An explicit config override must still mark the stage stale. "
        f"Got stale={stale!r}"
    )
