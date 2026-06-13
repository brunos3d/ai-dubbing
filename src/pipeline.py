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
from .utils.cache import CacheManager
from .utils.checkpoint import Checkpoint
from .utils.logging import get_logger, setup_logging, stage_banner
from .utils.paths import env, log_dir, output_dir, working_dir

LOG = setup_logging("ai-dubbing")


class Pipeline:
    """Runs the 11 stages with checkpointing and cache management."""

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
        no_cache: bool = False,
        from_stage: Optional[str] = None,
        read_only_cache: bool = False,
        glossary_path: Optional[Path] = None,
        no_pyannote: bool = False,
    ):
        global LOG
        env()
        self.input_path = input_path
        self.source_language = source_language
        self.target_language = target_language
        self.output_dir = output_dir(output_path)
        self.whisper_model = whisper_model
        self.hf_token = hf_token
        self.target_lufs = target_lufs
        self.skip_video = skip_video
        self.no_cache = no_cache
        self.from_stage = from_stage
        self.read_only_cache = read_only_cache
        self.glossary_path = glossary_path
        self.no_pyannote = no_pyannote

        # Cache resolution
        self.cache_manager = CacheManager()
        self.cache_key = self.cache_manager.get_cache_key(input_path)
        
        if workdir is not None:
            self.workdir = Path(workdir).resolve()
        else:
            self.workdir = self.cache_manager.get_working_dir(self.cache_key)
        
        if self.no_cache and self.workdir.exists():
            LOG.info("Forcing cache rebuild (--no-cache); clearing workdir")
            shutil.rmtree(self.workdir)

        self.workdir.mkdir(parents=True, exist_ok=True)

        # Re-initialize logging to use the actual workdir
        LOG = setup_logging("ai-dubbing", logfile=log_dir(self.workdir) / "pipeline.log")

        LOG.info(f"Cache key: {self.cache_key}")
        LOG.info(f"Workdir  : {self.workdir}")
        
        # Initialize metadata if it's a new cache entry
        if not (self.workdir / "metadata.json").exists() and not self.read_only_cache:
            self.cache_manager.init_metadata(
                self.cache_key, 
                input_path, 
                {"whisper_model": whisper_model, "target_lufs": target_lufs}
            )

        self.checkpoint = Checkpoint.load(self.workdir)
        
        # If --from-stage is set, invalidate everything from that stage onwards
        if self.from_stage and self.checkpoint:
            LOG.info(f"Rebuilding pipeline from stage: {self.from_stage}")
            found = False
            for name, _ in self.STAGES:
                if name == self.from_stage:
                    found = True
                if found:
                    self.checkpoint.stages.pop(name, None)
            self.checkpoint.save()

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
            if not self.read_only_cache:
                self.checkpoint.save()

    def _build_stage(self, name: str):
        if name == "extract":
            return ExtractStage(self.workdir)
        if name == "separate":
            return SeparateStage(self.workdir)
        if name == "diarize":
            return DiarizeStage(self.workdir, hf_token=self.hf_token, no_pyannote=self.no_pyannote)
        if name == "samples":
            return SampleStage(self.workdir)
        if name == "transcribe":
            return TranscribeStage(self.workdir, model_size=self.whisper_model, source_language=self.source_language)
        if name == "translate":
            return TranslateStage(self.workdir, self.source_language, self.target_language, glossary_path=self.glossary_path)
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
            # Rehydrate ALL prior stages that contributed inputs but haven't
            # been touched in *this* run.  The previous "just hydrate the
            # previous stage" approach left later stages with stale context
            # (e.g. generate running without speaker_samples set).
            for prev_idx in range(idx):
                prev_name = self.STAGES[prev_idx][0]
                if not context.get(f"_stage_{prev_name}_hydrated"):
                    self._rehydrate_from_disk(context, prev_name)
                    context[f"_stage_{prev_name}_hydrated"] = True

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
                    # Always re-read from disk so speaker_samples, transcripts,
                    # and similar dynamic collections stay in sync with the
                    # files actually on disk (the cached ``result`` may be
                    # empty for resumed runs).
                    self._rehydrate_from_disk(context, name)
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
        """Set up the context with everything stage ``name`` will need on disk.

        Reads the artefacts produced by stage ``name`` (and any earlier
        stages whose outputs are still required) and stashes them in
        ``context`` under the keys that downstream stages expect.
        """
        import json

        workdir = self.workdir
        output_dir = self.output_dir
        LOG.info(f"rehydrating context for stage '{name}'")
        spk_dir = workdir / "speakers"
        speaker_samples: Dict[str, str] = (
            {p.stem: str(p) for p in spk_dir.glob("*.wav")}
            if spk_dir.exists()
            else {}
        )
        ref_transcripts: Dict[str, str] = (
            {
                p.parent.name: (p.parent / "transcript.txt").read_text(encoding="utf-8").strip()
                for p in (workdir / "speaker_profiles").glob("*/transcript.txt")
            }
            if (workdir / "speaker_profiles").exists()
            else {}
        )

        # Common inputs the pipeline carries forward
        context.setdefault("audio_path", str(workdir / "original_audio.wav"))
        context["speech_path"] = str(workdir / "speech.wav")
        context["background_path"] = str(workdir / "background.wav")
        context["segments_path"] = str(workdir / "segments.json")
        context["transcript_path"] = str(workdir / "transcript.json")
        context["translated_path"] = str(workdir / "translated_transcript.json")
        context["generated_dir"] = str(workdir / "generated_segments")
        context["manifest_path"] = str(
            workdir / "generated_segments" / "manifest.json"
        )
        context["aligned_dir"] = str(workdir / "aligned_segments")
        aligned_manifest = str(workdir / "aligned_manifest.json")
        context["aligned_manifest"] = aligned_manifest
        # ``manifest_path`` is read by reconstruct/mix/align so make sure
        # it points to the aligned manifest for everything past generate.
        if name in {"align", "reconstruct", "mix", "video"}:
            context["manifest_path"] = aligned_manifest
        spk_profiles_dir = workdir / "speaker_profiles"
        speaker_profiles: Dict[str, Any] = {}
        if spk_profiles_dir.exists():
            for spk_path in spk_profiles_dir.iterdir():
                if not spk_path.is_dir():
                    continue
                spk = spk_path.name
                speaker_profiles[spk] = {"profiles": {}}
                for prof_path in spk_path.iterdir():
                    if not prof_path.is_dir():
                        continue
                    meta_path = prof_path / "metadata.json"
                    if meta_path.exists():
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        speaker_profiles[spk]["profiles"][prof_path.name] = {
                            "reference": meta["reference_path"],
                            "transcript_text": (prof_path / "transcript.txt").read_text(encoding="utf-8").strip() if (prof_path / "transcript.txt").exists() else "",
                            "score": meta.get("score", 0.0),
                        }

        context["speaker_profiles"] = speaker_profiles
        context["speaker_samples"] = speaker_samples
        context["ref_transcripts"] = ref_transcripts
        context["reconstructed_path"] = str(output_dir / "reconstructed_speech.wav")
        context["final_path"] = str(output_dir / "final_audio.wav")
        context["final_video"] = str(output_dir / "final_video.mp4")

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
            context["speaker_profiles"] = result.get("speaker_profiles", {})
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
