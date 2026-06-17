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


def _next_speed(
    measured: float,
    slot: float,
    cur_speed: float,
    max_speed: float,
    tol: float,
) -> Optional[float]:
    """Next speaking-speed to try, or None when no further pass helps.

    Only *overruns* (speech longer than the slot, which bleed into the next
    turn) are corrected, by scaling speed up proportionally and clamping to
    ``max_speed`` for naturalness. Underruns are left as natural pauses.
    """
    if slot <= 0 or measured <= 0:
        return None
    if measured <= slot * (1.0 + tol):
        return None  # fits, or underruns -> accept
    target = min(cur_speed * (measured / slot), max_speed)
    if target <= cur_speed + 1e-3:
        return None  # already at the cap; cannot compress further
    return round(target, 3)


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
    """Synthesize ``text`` so it fits ``slot_s``, iterating on speed.

    Returns ``(audio, sample_rate, final_speed, duration_s, iterations)``.
    Timing is treated as a generation constraint: we measure the real output
    and raise OmniVoice's speed to compress overruns, rather than fixing them
    after the fact with a lossy time-stretch. The best (least-overrun)
    candidate is returned.
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

    speed = float(base_speed)
    audio = _synth(speed)
    dur = audio.shape[0] / sr
    best = (audio, speed, dur)
    iters = 1
    for _ in range(max_iters):
        ns = _next_speed(dur, slot_s, speed, max_speed, tol)
        if ns is None:
            break
        speed = ns
        audio = _synth(speed)
        dur = audio.shape[0] / sr
        iters += 1
        # Track the candidate with the smallest overrun beyond the slot.
        if max(0.0, dur - slot_s) < max(0.0, best[2] - slot_s):
            best = (audio, speed, dur)
    # If the loop ended on a worse candidate than one seen earlier, use best.
    if max(0.0, dur - slot_s) > max(0.0, best[2] - slot_s):
        audio, speed, dur = best
    return audio, sr, speed, dur, iters


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
    config_fields: List[str] = [
        "model_id", "target_language", "use_clone_prompt",
        "max_speed", "max_fit_iters",
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
        subdir: str | None = None,
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
        self.subdir = subdir

    def output_paths(self) -> List[Path]:
        return [self.out_dir]

    def _load_model(self):
        from omnivoice import OmniVoice

        Path(self.offload_dir).mkdir(parents=True, exist_ok=True)
        import os
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        try:
            import torch

            free, _ = torch.cuda.mem_get_info()
            free_gb = free / (1024 ** 3)
        except Exception:
            free_gb = 0.0
        if free_gb < 3.5:
            gpu_cap = min(0.9, max(0.6, free_gb - 1.7))
            LOG.info(
                f"Only {free_gb:.2f} GB free VRAM; using device_map='auto' with offload "
                f"(GPU cap {gpu_cap:.1f} GiB for the LLM; audio tokenizer stays on GPU)"
            )
            try:
                return OmniVoice.from_pretrained(
                    self.model_id,
                    device_map="auto",
                    max_memory={0: f"{gpu_cap:.1f}GiB", "cpu": "30GiB"},
                    dtype=torch.float16,
                    offload_folder=self.offload_dir,
                )
            except Exception as exc:
                LOG.warning(f"auto device map with max_memory failed: {exc}")
        return OmniVoice.from_pretrained(
            self.model_id,
            device_map=self.device,
            dtype=torch.float16,
        )

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        stage_banner(LOG, 7, 11, "OmniVoice Generation")

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
                tts_model_id=self.model_id,
                slot_duration_s=slot,
                prosody_signature=str(seg.get("prosody_signature", "")),
            )

        manifest: List[Dict[str, Any]] = []
        model = None
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

            # --- Stable Identity Generation (lazy-load the TTS model) ---
            if model is None:
                LOG.info(f"Loading OmniVoice ({self.model_id}) on {self.device}")
                model = self._load_model()
                model_sr = int(getattr(model, "sampling_rate", _DEFAULT_SR))

            # We always use the 'primary' profile for each speaker for stability.
            prof_data = speaker_profiles[speaker]["profiles"]["primary"]
            ref_audio_path = prof_data["reference"]
            ref_transcript = ref_transcripts.get(speaker, "")

            LOG.info(
                f"Segment {i + 1:>3}: speaker={speaker} profile=primary "
                f"dur={orig_duration:.2f}s text={text[:60]!r}"
            )

            wav, sr = _coerce_ref_audio(ref_audio_path)
            # Honour a prosody-derived speed when present (Phase 5); default 1.0.
            speed = float(seg.get("prosody_speed", 1.0) or 1.0)

            # Timing as a generation constraint (Objective #3): synthesize,
            # measure, and raise speed to compress overruns until the line fits
            # its slot — instead of conditioning on the raw slot duration (which
            # OmniVoice fills by slowing the voice then silence-trimming) and
            # patching up afterwards with a lossy time-stretch.
            try:
                audio, _sr, fit_speed, gen_duration, fit_iters = generate_fitted(
                    model,
                    text=text,
                    language=_lang_name(self.target_language),
                    ref_audio=(wav, sr),
                    ref_text=ref_transcript or None,
                    slot_s=float(orig_duration),
                    base_speed=speed,
                    tol=self.duration_tolerance,
                    max_speed=self.max_speed,
                    max_iters=self.max_fit_iters,
                )
            except Exception as exc:  # noqa: BLE001
                LOG.warning(f"Segment {i + 1} fitted synthesis failed, falling back: {exc}")
                audios = model.generate(
                    text=text,
                    language=_lang_name(self.target_language),
                    ref_audio=(wav, sr),
                    ref_text=ref_transcript or None,
                )
                if not audios:
                    raise RuntimeError(f"OmniVoice produced no audio for segment {i + 1}")
                audio = audios[0]
                if isinstance(audio, torch.Tensor):
                    audio = audio.detach().cpu().numpy()
                audio = np.asarray(audio, dtype=np.float32).reshape(-1)
                fit_speed, fit_iters = speed, 1
                gen_duration = float(audio.shape[0]) / model_sr

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

        try:
            del model
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
