"""Stage 7 - Voice generation with OmniVoice."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch

from ..timing.prosody import analyze_segment, compute_gain, compute_speed
from ..tts import SpeakerProfile, make_synthesizer
from ..tts.base import fit_by_speed, next_speed as _next_speed  # re-export (compat)
from ..utils.audio import read_wav, write_wav
from ..utils.logging import get_logger, stage_banner
from ..utils.vram import free_vram, log_vram
from ..workspace.atomic import sha256_file
from ..workspace.timeline import segment_render_key

# Canonical sample rate for preserved (non-speech) and reused clips, used
# before the (lazily-loaded) TTS model is available.  OmniVoice's own output
# rate (usually also 24 kHz) is used for freshly synthesized segments.
_DEFAULT_SR = 24000

LOG = get_logger("ai-dubbing.generate")


LANGUAGE_NAME = {
    "en": "English",
    "pt": "Portuguese",
    "pt-br": "Portuguese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ru": "Russian",
}


def _lang_name(code: str) -> str:
    code = (code or "auto").lower()
    if code in LANGUAGE_NAME:
        return LANGUAGE_NAME[code]
    if "-" in code:
        short = code.split("-")[0]
        if short in LANGUAGE_NAME:
            return LANGUAGE_NAME[short]
    return code


def _coerce_ref_audio(path: str) -> tuple:
    """Read reference audio into a (waveform_tensor, sample_rate) tuple."""
    data, sr = read_wav(path, dtype="float32", always_2d=True)
    if data.shape[0] > 1:
        data = data.mean(axis=0, keepdims=True)
    wav = torch.from_numpy(data.astype("float32"))
    return wav, int(sr)


def _transcribe_ref_with_whisper(
    ref_audio_path: str,
    source_language: Optional[str] = None,
    model: Optional[Any] = None,
) -> str:
    """Transcribe the reference audio using faster-whisper to obtain a clean
    transcript.  This transcript is then passed as ``ref_text`` to OmniVoice so
    it doesn't have to load its own (GPU-hungry) ASR model.
    """
    from faster_whisper import WhisperModel

    local_model = False
    if model is None:
        from ..utils.asr import resolve_ref_asr
        size, device, compute = resolve_ref_asr()
        LOG.info(f"Transcribing reference audio with faster-whisper ({size} on {device}) -> {ref_audio_path}")
        try:
            model = WhisperModel(size, device=device, compute_type=compute)
            local_model = True
        except Exception as exc:
            LOG.warning(f"Failed to load whisper tiny for reference: {exc}")
            return "Reference audio for voice cloning."

    try:
        lang = source_language if source_language and source_language != "auto" else None
        segments, _ = model.transcribe(
            ref_audio_path, language=lang, beam_size=1, vad_filter=True,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
    except Exception as exc:
        LOG.warning(f"Whisper transcription of {ref_audio_path} failed: {exc}")
        text = ""
    finally:
        if local_model:
            try:
                del model
            except Exception:
                pass
            free_vram()
    if not text:
        text = "Reference audio for voice cloning."
    return text


def generate_fitted(
    model,
    text: str,
    language: str,
    ref_audio: tuple,
    ref_text: Optional[str],
    slot_s: float,
    base_speed: float = 1.0,
    tol: float = 0.10,
    max_speed: float = 1.35,
    max_iters: int = 2,
):
    """Compat shim over :func:`src.tts.base.fit_by_speed` for an OmniVoice model.

    Returns ``(audio, sample_rate, final_speed, duration_s, iterations)``. The
    canonical fitting loop now lives in the backend-agnostic ``tts`` layer; this
    wrapper keeps the OmniVoice-model call shape used by older callers/tests.
    """
    sr = int(getattr(model, "sampling_rate", _DEFAULT_SR))

    def _synth(speed: float) -> np.ndarray:
        audios = model.generate(
            text=text, language=language, ref_audio=ref_audio,
            ref_text=ref_text, speed=float(speed),
        )
        if not audios:
            raise RuntimeError("OmniVoice returned no audio")
        a = audios[0]
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
        return np.asarray(a, dtype=np.float32).reshape(-1)

    seg = fit_by_speed(
        _synth, sr, slot_s, base_speed=base_speed,
        tol=tol, max_speed=max_speed, max_iters=max_iters,
    )
    return seg.samples, seg.sample_rate, seg.speed, seg.duration_s, seg.iterations


def _generate_one(
    model,
    text: str,
    language: str,
    ref_audio_path: str,
    ref_text: Optional[str] = None,
    target_duration: Optional[float] = None,
    speed: float = 1.0,
) -> np.ndarray:
    wav, sr = _coerce_ref_audio(ref_audio_path)
    kwargs: Dict[str, Any] = {
        "text": text,
        "language": _lang_name(language),
        "ref_audio": (wav, sr),
        "ref_text": ref_text,
        "speed": speed,
    }
    if target_duration is not None and target_duration > 0:
        kwargs["duration"] = float(target_duration)
    audios = model.generate(**kwargs)
    if not audios:
        raise RuntimeError("OmniVoice returned no audio")
    audio = audios[0]
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


class GenerateStage:
    name = "generate"
    inputs: List[str] = ["translation/translated_transcript.json"]
    outputs: List[str] = [
        "generated_segments/manifest.json",
        "generated_segments/prosody.json",
    ]
    editable_outputs: List[str] = []
    derived_outputs: List[str] = [
        "generated_segments/manifest.json",
        "generated_segments/prosody.json",
    ]
    # ``tts_backend`` is part of the config so switching it invalidates this
    # stage (and, via the dependency DAG, align/reconstruct/mix/video) without
    # touching extract..translate.
    config_fields: List[str] = [
        "model_id", "target_language", "use_clone_prompt",
        "max_speed", "max_fit_iters", "tts_backend",
        "num_step", "denoise", "guidance_scale", "t_shift",
        "position_temperature", "class_temperature",
        "preprocess_prompt", "postprocess_output", "tail_padding_s",
        "audio_chunk_duration", "audio_chunk_threshold",
    ]

    def __init__(
        self,
        workdir: Path,
        model_id: str = "k2-fsa/OmniVoice",
        target_language: str = "en",
        device: str = "cuda",
        out_dir_name: str = "generated_segments",
        use_clone_prompt: bool = True,
        duration_tolerance: float = 0.10,
        max_speed: float = 1.35,
        max_fit_iters: int = 2,
        offload_dir: str = "/tmp/opencode/omnivoice_offload",
        tts_backend: str = "omnivoice",
        f5_model: str = "F5TTS_v1_Base",
        f5_ckpt_file: str = "",
        f5_vocab_file: str = "",
        subdir: str | None = None,
        # New OmniVoice parameters
        num_step: int = 32,
        denoise: bool = True,
        guidance_scale: float = 2.0,
        t_shift: float = 0.1,
        position_temperature: float = 5.0,
        class_temperature: float = 0.0,
        preprocess_prompt: bool = True,
        postprocess_output: bool = True,
        tail_padding_s: float = 0.1,
        audio_chunk_duration: float = 15.0,
        audio_chunk_threshold: float = 30.0,
    ):
        self.workdir = Path(workdir)
        if subdir:
            self.workdir = self.workdir / subdir
        self.model_id = model_id
        self.target_language = target_language
        self.device = device
        self.out_dir = self.workdir / out_dir_name
        self.use_clone_prompt = use_clone_prompt
        self.duration_tolerance = duration_tolerance
        # Iterative duration-fitting bounds (Objective #3).
        self.max_speed = max_speed
        self.max_fit_iters = max_fit_iters
        self.offload_dir = offload_dir
        # Pluggable TTS backend (default: OmniVoice).
        self.tts_backend = (tts_backend or "omnivoice").strip().lower()
        self.f5_model = f5_model
        self.f5_ckpt_file = f5_ckpt_file
        self.f5_vocab_file = f5_vocab_file
        self.subdir = subdir
        
        # New OmniVoice parameters
        self.num_step = num_step
        self.denoise = denoise
        self.guidance_scale = guidance_scale
        self.t_shift = t_shift
        self.position_temperature = position_temperature
        self.class_temperature = class_temperature
        self.preprocess_prompt = preprocess_prompt
        self.postprocess_output = postprocess_output
        self.tail_padding_s = tail_padding_s
        self.audio_chunk_duration = audio_chunk_duration
        self.audio_chunk_threshold = audio_chunk_threshold

    def _make_synthesizer(self):
        """Build the selected :class:`VoiceSynthesizer` (model loads lazily)."""
        kwargs: Dict[str, Any] = {
            "language": self.target_language,
            "device": self.device,
            "tolerance": self.duration_tolerance,
            "max_speed": self.max_speed,
            "max_fit_iters": self.max_fit_iters,
        }
        if self.tts_backend == "omnivoice":
            kwargs.update({
                "model_id": self.model_id,
                "offload_dir": self.offload_dir,
                "num_step": self.num_step,
                "denoise": self.denoise,
                "guidance_scale": self.guidance_scale,
                "t_shift": self.t_shift,
                "position_temperature": self.position_temperature,
                "class_temperature": self.class_temperature,
                "preprocess_prompt": self.preprocess_prompt,
                "postprocess_output": self.postprocess_output,
                "tail_padding_s": self.tail_padding_s,
                "audio_chunk_duration": self.audio_chunk_duration,
                "audio_chunk_threshold": self.audio_chunk_threshold,
            })
        elif self.tts_backend == "f5tts":
            kwargs["model"] = self.f5_model
            if self.f5_ckpt_file:
                kwargs["ckpt_file"] = self.f5_ckpt_file
            if self.f5_vocab_file:
                kwargs["vocab_file"] = self.f5_vocab_file
        return make_synthesizer(self.tts_backend, **kwargs)

    def _tts_tag(self) -> str:
        """Backend-qualified model id for the per-segment render key."""
        if self.tts_backend == "f5tts":
            return f"f5tts:{self.f5_model}"
        return f"{self.tts_backend}:{self.model_id}"

    def output_paths(self) -> List[Path]:
        return [self.out_dir]

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        stage_banner(LOG, 7, 11, f"Voice Generation ({self.tts_backend})")

        transcript_path = Path(context["translated_path"])
        speech_path = Path(context["speech_path"])
        full_audio, full_sr = read_wav(speech_path)
        if full_audio.ndim == 2:
            full_audio = full_audio.mean(axis=0)

        speaker_profiles: Dict[str, Any] = context.get("speaker_profiles", {})
        segments = json.loads(transcript_path.read_text(encoding="utf-8"))
        if not segments:
            raise RuntimeError("No translated segments to synthesize")

        speakers = list(speaker_profiles.keys())
        if not speakers:
            raise RuntimeError("No speaker voice profiles available")

        self.out_dir.mkdir(parents=True, exist_ok=True)
        source_language = context.get("source_language")
        
        # Resolve reference transcripts.  The samples stage already wrote a
        # ``transcript.txt`` per speaker profile (surfaced here as
        # ``transcript_text``), so in the common case no ASR is needed at
        # all.  whisper-tiny is loaded *lazily* — only if some profile is
        # missing its transcript — which avoids a redundant model load on
        # every generate run (previously loaded unconditionally).
        ref_transcripts: Dict[str, str] = {}
        missing = [
            spk for spk, d in speaker_profiles.items()
            if not d["profiles"]["primary"].get("transcript_text")
        ]
        whisper_tiny = None
        if missing:
            from faster_whisper import WhisperModel
            from ..utils.asr import resolve_ref_asr
            size, device, compute = resolve_ref_asr()
            try:
                whisper_tiny = WhisperModel(size, device=device, compute_type=compute)
            except Exception as exc:
                LOG.warning(f"Could not load reference ASR ({size}/{device}): {exc}; trying tiny/cpu")
                try:
                    whisper_tiny = WhisperModel("tiny", device="cpu", compute_type="int8")
                except Exception:
                    whisper_tiny = None
        else:
            LOG.info("All speaker references already transcribed; skipping reference-ASR load")

        try:
            for spk, spk_data in speaker_profiles.items():
                p = spk_data["profiles"]["primary"]
                if p.get("transcript_text"):
                    ref_transcripts[spk] = p["transcript_text"]
                else:
                    try:
                        ref_transcripts[spk] = _transcribe_ref_with_whisper(
                            p["reference"], source_language, model=whisper_tiny
                        )
                    except Exception as exc:  # noqa: BLE001
                        LOG.warning(f"Could not transcribe {spk} reference: {exc}")
                        ref_transcripts[spk] = ""
        finally:
            if whisper_tiny:
                try:
                    del whisper_tiny
                except Exception:
                    pass
                free_vram()

        # --- Prosody analysis (spec Phase 5) -------------------------------
        # Pull cheap delivery cues out of the source speech so synthesis can
        # match tempo (TTS speed) and reconstruction can match relative
        # loudness (per-segment gain).  Descriptors fold into the render key
        # so a prosody change re-renders the segment.
        src_lang = context.get("source_language")
        prosody_by_index: Dict[int, Any] = {}
        spk_rates: Dict[str, List[float]] = {}
        spk_energy: Dict[str, List[float]] = {}
        for i, seg in enumerate(segments):
            if seg.get("is_non_speech"):
                continue
            d = analyze_segment(
                full_audio, full_sr,
                float(seg.get("start", 0.0)), float(seg.get("end", 0.0)),
                source_text=seg.get("source_text") or seg.get("text"),
                lang=src_lang,
            )
            prosody_by_index[i] = d
            spk = seg.get("speaker", "")
            if d.speaking_rate_syl_s:
                spk_rates.setdefault(spk, []).append(d.speaking_rate_syl_s)
            if d.energy_rms > 0:
                spk_energy.setdefault(spk, []).append(d.energy_rms)

        def _mean(vals: List[float]) -> Optional[float]:
            return float(np.median(vals)) if vals else None

        mean_rate = {s: _mean(v) for s, v in spk_rates.items()}
        mean_energy = {s: _mean(v) for s, v in spk_energy.items()}
        for i, seg in enumerate(segments):
            d = prosody_by_index.get(i)
            if d is None:
                continue
            spk = seg.get("speaker", "")
            seg["prosody_speed"] = compute_speed(d.speaking_rate_syl_s, mean_rate.get(spk))
            seg["prosody_signature"] = d.signature()
            seg["_prosody_gain"] = compute_gain(d.energy_rms, mean_energy.get(spk) or 0.0)

        # --- Segment-level reuse (spec Phase 4) ----------------------------
        # A per-speaker voice hash (cheap: one hash per reference) plus the
        # previous run's manifest let us compute a stable render_key per
        # segment and skip re-synthesizing any segment whose inputs are
        # unchanged.  Editing one translated line re-renders one segment;
        # everything else is reused.  If *nothing* needs synthesizing the
        # OmniVoice model is never loaded at all.
        voice_hash: Dict[str, str] = {}
        for spk, d in speaker_profiles.items():
            ref = d["profiles"]["primary"].get("reference")
            try:
                voice_hash[spk] = sha256_file(Path(ref))[:16] if ref else ""
            except OSError:
                voice_hash[spk] = ""

        prev_by_key: Dict[str, Dict[str, Any]] = {}
        prev_manifest_path = self.out_dir / "manifest.json"
        if prev_manifest_path.exists():
            try:
                for e in json.loads(prev_manifest_path.read_text(encoding="utf-8")):
                    rk, p = e.get("render_key"), e.get("path")
                    if rk and p and Path(p).exists():
                        prev_by_key[rk] = e
            except (json.JSONDecodeError, OSError):
                prev_by_key = {}

        def _render_key(seg: Dict[str, Any], speaker: str) -> str:
            txt = (seg.get("text") or "").strip()
            marker = ("[nonspeech]" + txt) if seg.get("is_non_speech") else txt
            slot = float(seg.get("end", 0.0)) - float(seg.get("start", 0.0))
            return segment_render_key(
                target_text=marker,
                speaker=speaker,
                target_language=self.target_language,
                voice_profile_hash=voice_hash.get(speaker, ""),
                tts_model_id=self._tts_tag(),
                slot_duration_s=slot,
                prosody_signature=str(seg.get("prosody_signature", "")),
            )

        manifest: List[Dict[str, Any]] = []
        synth = None  # VoiceSynthesizer, loaded lazily on first speech segment
        model_sr = _DEFAULT_SR
        reused_count = 0
        synth_count = 0

        for i, seg in enumerate(segments):
            text = (seg.get("text") or "").strip()
            speaker = seg.get("speaker", speakers[0])
            if speaker not in speaker_profiles:
                speaker = speakers[0]
            start_s = float(seg.get("start", 0.0))
            end_s = float(seg.get("end", 0.0))
            orig_duration = end_s - start_s
            out_wav = self.out_dir / f"segment_{i + 1:04d}.wav"
            rkey = _render_key(seg, speaker)

            # --- Reuse an unchanged segment from a previous run ---
            reuse = prev_by_key.get(rkey)
            if reuse is not None:
                prior = Path(reuse["path"])
                if prior.resolve() != out_wav.resolve() and prior.exists():
                    shutil.copy2(prior, out_wav)
                entry = dict(reuse)
                entry.update({
                    "index": i, "speaker": speaker, "start": start_s, "end": end_s,
                    "path": str(out_wav), "render_key": rkey, "reused": True,
                })
                manifest.append(entry)
                reused_count += 1
                LOG.info(f"Segment {i + 1:>3}: REUSED (render_key {rkey})")
                continue

            # --- Non-Speech Preservation ---
            if seg.get("is_non_speech"):
                LOG.info(f"Segment {i + 1:>3}: PRESERVING ORIGINAL (non-speech event: {text})")
                s0 = int(start_s * full_sr)
                s1 = int(end_s * full_sr)
                clip = full_audio[s0:s1]
                if full_sr != _DEFAULT_SR:
                    from ..utils.audio import resample
                    clip = resample(clip, full_sr, _DEFAULT_SR)
                sf.write(str(out_wav), clip, _DEFAULT_SR, subtype="PCM_16")
                manifest.append({
                    "index": i,
                    "speaker": speaker,
                    "is_non_speech": True,
                    "text": text,
                    "start": start_s,
                    "end": end_s,
                    "original_duration": round(orig_duration, 3),
                    "generated_duration": round(len(clip) / _DEFAULT_SR, 3),
                    "path": str(out_wav),
                    "sample_rate": _DEFAULT_SR,
                    "render_key": rkey,
                    "reused": False,
                })
                continue

            # --- Stable Identity Generation (lazy-load the TTS backend) ---
            if synth is None:
                synth = self._make_synthesizer()
                synth.load()
                model_sr = int(synth.sample_rate)

            # We always use the 'primary' profile for each speaker for stability.
            prof_data = speaker_profiles[speaker]["profiles"]["primary"]
            profile = SpeakerProfile(
                speaker_id=speaker,
                reference_wav=prof_data["reference"],
                reference_text=ref_transcripts.get(speaker, "") or None,
            )

            LOG.info(
                f"Segment {i + 1:>3}: speaker={speaker} profile=primary "
                f"dur={orig_duration:.2f}s text={text[:60]!r}"
            )

            # Honour a prosody-derived speed when present (Phase 5); default 1.0.
            speed = float(seg.get("prosody_speed", 1.0) or 1.0)

            # Timing as a generation constraint (Objective #3): the backend
            # synthesizes, measures, and raises speed to fit the slot. This is
            # backend-agnostic — OmniVoice and F5-TTS both go through the same
            # VoiceSynthesizer contract.
            result = synth.generate(
                text=text,
                speaker_profile=profile,
                target_duration=float(orig_duration),
                base_speed=speed,
            )
            audio = result.samples
            model_sr = int(result.sample_rate)
            gen_duration = result.duration_s
            fit_speed = result.speed
            fit_iters = result.iterations

            if fit_iters > 1:
                LOG.info(
                    f"  fitted in {fit_iters} passes: speed={fit_speed:.2f} "
                    f"dur={gen_duration:.2f}s slot={orig_duration:.2f}s"
                )
            speed = fit_speed
            sf.write(str(out_wav), audio, model_sr, subtype="PCM_16")

            _pros = prosody_by_index.get(i)
            manifest.append({
                "index": i,
                "speaker": speaker,
                "is_non_speech": False,
                "text": text,
                "source_text": seg.get("source_text", ""),
                "start": start_s,
                "end": end_s,
                "target_duration": round(orig_duration, 3),
                "original_duration": round(orig_duration, 3),
                "generated_duration": round(gen_duration, 3),
                "path": str(out_wav),
                "sample_rate": model_sr,
                "render_key": rkey,
                "reused": False,
                "tts_backend": self.tts_backend,
                "prosody_speed": float(seg.get("prosody_speed", 1.0) or 1.0),
                "prosody_gain": float(seg.get("_prosody_gain", 1.0) or 1.0),
                "prosody": _pros.to_dict() if _pros is not None else {},
            })
            synth_count += 1

            if (i + 1) % 5 == 0 or i == len(segments):
                LOG.info(f"Generated {i + 1}/{len(segments)} segments")

        LOG.info(
            f"Generation plan: synthesized={synth_count} reused={reused_count} "
            f"total={len(segments)}"
        )
        manifest_path = self.out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

        # Prosody report (derived diagnostic; also surfaced on the Timeline).
        prosody_report = {
            "$schema_version": 1,
            "segments": [
                {
                    "index": i,
                    "speaker": segments[i].get("speaker"),
                    "prosody_speed": float(segments[i].get("prosody_speed", 1.0) or 1.0),
                    "prosody_gain": float(segments[i].get("_prosody_gain", 1.0) or 1.0),
                    **prosody_by_index[i].to_dict(),
                }
                for i in sorted(prosody_by_index)
            ],
        }
        prosody_path = self.out_dir / "prosody.json"
        prosody_path.write_text(json.dumps(prosody_report, indent=2, ensure_ascii=False))

        if synth is not None:
            try:
                synth.close()
            except Exception:
                pass
        free_vram()
        log_vram(LOG)
        return {
            "generated_dir": str(self.out_dir),
            "manifest_path": str(manifest_path),
            "prosody_path": str(prosody_path),
            "ref_transcripts": ref_transcripts,
        }
