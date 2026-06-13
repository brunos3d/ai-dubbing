"""Pipeline orchestrator with checkpointing."""
from __future__ import annotations

import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from .stages import (
    AlignStage,
    DiarizeStage,
    ExtractStage,
    GenerateStage,
    MixStage,
    ReconstructStage,
    SampleStage,
    SeparateStage,
    TranslateStage,
    TranscribeStage,
    VideoStage,
)
from .utils.checkpoint import Checkpoint
from .utils.logging import get_logger, setup_logging, stage_banner
from .utils.paths import env, log_dir, output_dir, working_dir

LOG = setup_logging("ai-dubbing")


class Pipeline:
    """Runs the 11 stages with checkpointing."""

    STAGES = [
        ("extract", ExtractStage),
        ("separate", SeparateStage),
        ("diarize", DiarizeStage),
        ("samples", SampleStage),
        ("transcribe", TranscribeStage),
        ("translate", TranslateStage),
        ("generate", GenerateStage),
        ("align", AlignStage),
        ("reconstruct", ReconstructStage),
        ("mix", MixStage),
        ("video", VideoStage),
    ]

    def __init__(
        self,
        input_path: str,
        source_language: str,
        target_language: str,
        workdir: Optional[Path] = None,
        output_path: Optional[Path] = None,
        whisper_model: str = "large-v3",
        hf_token: Optional[str] = None,
        target_lufs: float = -16.0,
        skip_video: bool = False,
    ):
        env()
        self.input_path = input_path
        self.source_language = source_language
        self.target_language = target_language
        self.workdir = working_dir(workdir)
        self.output_dir = output_dir(output_path)
        self.whisper_model = whisper_model
        self.hf_token = hf_token
        self.target_lufs = target_lufs
        self.skip_video = skip_video

        self.checkpoint = Checkpoint.load(self.workdir)
        same_job = (
            self.checkpoint is not None
            and self.checkpoint.input_path == input_path
            and self.checkpoint.source_language == source_language
            and self.checkpoint.target_language == target_language
        )
        if not same_job:
            self.checkpoint = Checkpoint.create(
                input_path=input_path,
                source_language=source_language,
                target_language=target_language,
                workdir=str(self.workdir),
                output_dir=str(self.output_dir),
                config={"whisper_model": whisper_model, "target_lufs": target_lufs},
            )
            self.checkpoint.save()

    def _build_stage(self, name: str):
        if name == "extract":
            return ExtractStage(self.workdir)
        if name == "separate":
            return SeparateStage(self.workdir)
        if name == "diarize":
            return DiarizeStage(self.workdir, hf_token=self.hf_token)
        if name == "samples":
            return SampleStage(self.workdir)
        if name == "transcribe":
            return TranscribeStage(self.workdir, model_size=self.whisper_model, source_language=self.source_language)
        if name == "translate":
            return TranslateStage(self.workdir, self.source_language, self.target_language)
        if name == "generate":
            return GenerateStage(self.workdir, target_language=self.target_language)
        if name == "align":
            return AlignStage(self.workdir, target_language=self.target_language)
        if name == "reconstruct":
            return ReconstructStage(self.workdir, self.output_dir)
        if name == "mix":
            return MixStage(self.workdir, self.output_dir, target_lufs=self.target_lufs)
        if name == "video":
            return VideoStage(self.workdir, self.output_dir, self.input_path)
        raise KeyError(name)

    def run(self, start_from: str = "extract", only: Optional[str] = None) -> Dict[str, Any]:
        start_idx = next((i for i, (n, _) in enumerate(self.STAGES) if n == start_from), 0)
        context: Dict[str, Any] = {
            "input_path": self.input_path,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "workdir": str(self.workdir),
            "output_dir": str(self.output_dir),
        }
        names = [n for n, _ in self.STAGES]
        if only is not None:
            indices = [names.index(only)]
        else:
            indices = list(range(start_idx, len(self.STAGES)))

        for idx in indices:
            name, _ = self.STAGES[idx]
            if self.skip_video and name == "video":
                LOG.info("Skipping video stage (--no-video / --audio-only)")
                continue
            if idx > 0 and not context.get(f"_stage_{self.STAGES[idx - 1][0]}_hydrated"):
                self._rehydrate_from_disk(context, self.STAGES[idx - 1][0])
            if self.checkpoint.is_done(name):
                # Verify the recorded outputs still exist on disk; otherwise
                # treat the cache as stale and re-run the stage.
                rec = self.checkpoint.stages.get(name)
                outputs = (rec.output_files if rec else []) or []
                missing = [p for p in outputs if not Path(p).exists()]
                if missing:
                    LOG.warning(
                        f"[{idx + 1}/{len(self.STAGES)}] {name} cache is stale "
                        f"({len(missing)} output file(s) missing); re-running"
                    )
                    self.checkpoint.stages.pop(name, None)
                else:
                    LOG.info(f"[{idx + 1}/{len(self.STAGES)}] {name} (cached)")
                    self._hydrate_context(context, name)
                    context[f"_stage_{name}_hydrated"] = True
                    continue
            stage = self._build_stage(name)
            self.checkpoint.mark_running(name)
            t0 = time.time()
            try:
                stage_banner(LOG, idx, len(self.STAGES), name.replace('_', ' ').title())
                if name == "extract":
                    result = stage.run(self.input_path, context)
                else:
                    LOG.debug(f"Running {name} with context keys: {list(context.keys())}")
                    if name in ("reconstruct", "mix"):
                        LOG.info(f"context['manifest_path']={context.get('manifest_path')}")
                    result = stage.run(context)
                outputs = [str(p) for p in stage.outputs() if Path(p).exists()]
                self.checkpoint.mark_done(name, output_files=outputs, metadata={"duration_s": time.time() - t0})
                self._hydrate_context(context, name, result)
                context[f"_stage_{name}_hydrated"] = True
            except Exception as exc:  # noqa: BLE001
                self.checkpoint.mark_failed(name, str(exc))
                LOG.error(f"Stage {name} failed: {exc}")
                LOG.debug(traceback.format_exc())
                raise

        return context

    def _rehydrate_from_disk(self, context: Dict[str, Any], name: str) -> None:
        """When starting from a mid-pipeline stage, set up the context with
        the artefacts produced by the named previous stage.
        """
        import json

        workdir = self.workdir
        LOG.info(f"rehydrating context from previous stage '{name}'")
        spk_dir = workdir / "speakers"
        speaker_samples: Dict[str, str] = (
            {p.stem: str(p) for p in spk_dir.glob("*.wav")}
            if spk_dir.exists()
            else {}
        )
        if name == "extract":
            context["audio_path"] = str(workdir / "original_audio.wav")
        elif name == "separate":
            context["audio_path"] = str(workdir / "original_audio.wav")
            context["speech_path"] = str(workdir / "speech.wav")
            context["background_path"] = str(workdir / "background.wav")
        elif name == "diarize":
            context["speech_path"] = str(workdir / "speech.wav")
            context["background_path"] = str(workdir / "background.wav")
            context["segments_path"] = str(workdir / "segments.json")
        elif name == "samples":
            context["speech_path"] = str(workdir / "speech.wav")
            context["segments_path"] = str(workdir / "segments.json")
            context["speaker_samples"] = speaker_samples
        elif name == "transcribe":
            context["speech_path"] = str(workdir / "speech.wav")
            context["segments_path"] = str(workdir / "segments.json")
            context["speaker_samples"] = speaker_samples
        elif name == "translate":
            context["transcript_path"] = str(workdir / "transcript.json")
        elif name == "generate":
            context["translated_path"] = str(workdir / "translated_transcript.json")
            context["generated_dir"] = str(workdir / "generated_segments")
            context["manifest_path"] = str(
                workdir / "generated_segments" / "manifest.json"
            )
            context["speaker_samples"] = speaker_samples
        elif name == "align":
            context["generated_dir"] = str(workdir / "generated_segments")
            context["aligned_dir"] = str(workdir / "aligned_segments")
            aligned_manifest = str(workdir / "aligned_manifest.json")
            context["aligned_manifest"] = aligned_manifest
            context["manifest_path"] = aligned_manifest
            context["background_path"] = str(workdir / "background.wav")
            context["speaker_samples"] = speaker_samples
        elif name == "reconstruct":
            context["manifest_path"] = str(workdir / "aligned_manifest.json")
            context["aligned_manifest"] = str(workdir / "aligned_manifest.json")
            context["background_path"] = str(workdir / "background.wav")
            context["speech_path"] = str(workdir / "speech.wav")
            context["reconstructed_path"] = str(
                self.output_dir / "reconstructed_speech.wav"
            )
        elif name == "mix":
            recon = workdir / "reconstructed_speech.wav"
            if not recon.exists():
                recon = self.output_dir / "reconstructed_speech.wav"
            context["reconstructed_path"] = str(recon)
            context["background_path"] = str(workdir / "background.wav")
            context["speech_path"] = str(workdir / "speech.wav")
            context["final_path"] = str(self.output_dir / "final_audio.wav")
        elif name == "video":
            context["final_path"] = str(self.output_dir / "final_audio.wav")

        return context

    def _hydrate_context(self, context: Dict[str, Any], name: str, result: Optional[Dict[str, Any]] = None) -> None:
        if result is None:
            result = {}
        if name == "extract":
            context["audio_path"] = result.get("audio_path") or str(self.workdir / "original_audio.wav")
            context["duration_s"] = result.get("duration_s")
        elif name == "separate":
            context["speech_path"] = result.get("speech_path") or str(self.workdir / "speech.wav")
            context["background_path"] = result.get("background_path") or str(self.workdir / "background.wav")
        elif name == "diarize":
            context["segments_path"] = result.get("segments_path") or str(self.workdir / "segments.json")
            context["speakers"] = result.get("speakers", [])
        elif name == "samples":
            context["speaker_samples"] = result.get("speaker_samples", {})
            context["speakers_dir"] = result.get("speakers_dir", str(self.workdir / "speakers"))
        elif name == "transcribe":
            context["transcript_path"] = result.get("transcript_path") or str(self.workdir / "transcript.json")
            context["word_path"] = result.get("word_path", str(self.workdir / "transcript_word_level.json"))
        elif name == "translate":
            context["translated_path"] = result.get("translated_path") or str(self.workdir / "translated_transcript.json")
        elif name == "generate":
            context["generated_dir"] = result.get("generated_dir", str(self.workdir / "generated_segments"))
            context["manifest_path"] = result.get("manifest_path", str(self.workdir / "generated_segments" / "manifest.json"))
        elif name == "align":
            context["aligned_dir"] = result.get("aligned_dir", str(self.workdir / "aligned_segments"))
            aligned_manifest = result.get("manifest_path", str(self.workdir / "aligned_manifest.json"))
            context["aligned_manifest"] = aligned_manifest
            context["manifest_path"] = aligned_manifest
        elif name == "reconstruct":
            recon = result.get("reconstructed_path") or str(self.output_dir / "reconstructed_speech.wav")
            context["reconstructed_path"] = recon
        elif name == "mix":
            context["final_path"] = result.get("final_path") or str(self.output_dir / "final_audio.wav")
        elif name == "video":
            context["final_video"] = result.get("video_path") or str(self.output_dir / "final_video.mp4")
