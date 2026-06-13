"""Stage 7 - Voice generation with OmniVoice."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch

from ..utils.audio import read_wav, write_wav
from ..utils.logging import get_logger, stage_banner
from ..utils.vram import free_vram, log_vram

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
) -> str:
    """Transcribe the reference audio using faster-whisper to obtain a clean
    transcript.  This transcript is then passed as ``ref_text`` to OmniVoice so
    it doesn't have to load its own (GPU-hungry) ASR model.
    """
    from faster_whisper import WhisperModel

    LOG.info(f"Transcribing reference audio with faster-whisper (tiny) -> {ref_audio_path}")
    try:
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        lang = source_language if source_language and source_language != "auto" else None
        segments, _ = model.transcribe(
            ref_audio_path, language=lang, beam_size=1, vad_filter=True,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
    finally:
        try:
            del model
        except Exception:
            pass
        free_vram()
    if not text:
        text = "Reference audio for voice cloning."
    return text


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

    def __init__(
        self,
        workdir: Path,
        model_id: str = "k2-fsa/OmniVoice",
        target_language: str = "en",
        device: str = "cuda",
        out_dir_name: str = "generated_segments",
        use_clone_prompt: bool = True,
        duration_tolerance: float = 0.10,
        offload_dir: str = "/tmp/opencode/omnivoice_offload",
    ):
        self.workdir = Path(workdir)
        self.model_id = model_id
        self.target_language = target_language
        self.device = device
        self.out_dir = self.workdir / out_dir_name
        self.use_clone_prompt = use_clone_prompt
        self.duration_tolerance = duration_tolerance
        self.offload_dir = offload_dir

    def outputs(self) -> List[Path]:
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
        stage_banner(LOG, 7, 12, "OmniVoice Generation") # Updated total stages

        transcript_path = Path(context.get("use_transcript_path", context["translated_path"]))
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

        # --- Narrator Mode Activation ---
        is_narrator_mode = len(speakers) == 1
        if is_narrator_mode:
            LOG.info("Narrator Mode activated: Maintaining stable identity for single speaker.")

        self.out_dir.mkdir(parents=True, exist_ok=True)
        source_language = context.get("source_language")
        
        # Pre-transcribe ALL primary profiles
        ref_transcripts: Dict[str, str] = {}
        for spk, spk_data in speaker_profiles.items():
            p = spk_data["profiles"]["primary"]
            if p.get("transcript_text"):
                ref_transcripts[spk] = p["transcript_text"]
            else:
                try:
                    ref_transcripts[spk] = _transcribe_ref_with_whisper(p["reference"], source_language)
                except Exception as exc:  # noqa: BLE001
                    LOG.warning(f"Could not transcribe {spk} reference: {exc}")
                    ref_transcripts[spk] = ""
        free_vram()

        LOG.info(f"Loading OmniVoice ({self.model_id}) on {self.device}")
        model = self._load_model()
        sample_rate = int(getattr(model, "sampling_rate", 24000))

        manifest: List[Dict[str, Any]] = []
        for i, seg in enumerate(segments):
            text = (seg.get("text") or "").strip()
            speaker = seg.get("speaker", speakers[0])
            start_s = float(seg.get("start", 0.0))
            end_s = float(seg.get("end", 0.0))
            orig_duration = end_s - start_s
            
            if speaker not in speaker_profiles:
                speaker = speakers[0]

            out_wav = self.out_dir / f"segment_{i + 1:04d}.wav"
            
            # --- Non-Speech Preservation ---
            if seg.get("is_non_speech"):
                LOG.info(f"Segment {i + 1:>3}: PRESERVING ORIGINAL (non-speech event: {text})")
                s0 = int(start_s * full_sr)
                s1 = int(end_s * full_sr)
                clip = full_audio[s0:s1]
                if full_sr != sample_rate:
                    import librosa
                    clip = librosa.resample(clip.astype("float32"), orig_sr=full_sr, target_sr=sample_rate)
                sf.write(str(out_wav), clip, sample_rate, subtype="PCM_16")
                
                manifest.append({
                    "index": i,
                    "speaker": speaker,
                    "is_non_speech": True,
                    "text": text,
                    "start": start_s,
                    "end": end_s,
                    "original_duration": round(orig_duration, 3),
                    "generated_duration": round(len(clip) / sample_rate, 3),
                    "path": str(out_wav),
                })
                continue

            # --- Stable Identity Generation ---
            # We always use the 'primary' profile for each speaker to ensure stability.
            prof_data = speaker_profiles[speaker]["profiles"]["primary"]
            ref_audio_path = prof_data["reference"]
            ref_transcript = ref_transcripts.get(speaker, "")

            LOG.info(
                f"Segment {i + 1:>3}: speaker={speaker} profile=primary "
                f"dur={orig_duration:.2f}s text={text[:60]!r}"
            )

            wav, sr = _coerce_ref_audio(ref_audio_path)
            kwargs: Dict[str, Any] = {
                "text": text,
                "language": _lang_name(self.target_language),
                "ref_audio": (wav, sr),
                "ref_text": ref_transcript or None,
                "duration": float(orig_duration),
            }

            try:
                audios = model.generate(**kwargs)
            except Exception as exc:  # noqa: BLE001
                LOG.warning(f"Segment {i + 1} failed with duration conditioning, falling back: {exc}")
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
            sf.write(str(out_wav), audio, sample_rate, subtype="PCM_16")
            gen_duration = float(audio.shape[0]) / sample_rate

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
                "sample_rate": sample_rate,
            })

            if (i + 1) % 5 == 0 or i == len(segments):
                LOG.info(f"Generated {i + 1}/{len(segments)} segments")

        # --- Continuity Metrics ---
        short_segs = sum(1 for m in manifest if m["original_duration"] < 4.0 and not m["is_non_speech"])
        ideal_segs = sum(1 for m in manifest if 8.0 <= m["original_duration"] <= 12.0)
        total_speech = sum(1 for m in manifest if not m["is_non_speech"])
        
        # Penalize short segments and switches (switches is 0 in this architecture)
        continuity_score = 100.0
        if total_speech > 0:
            continuity_score -= (short_segs / total_speech) * 40.0
            continuity_score += (ideal_segs / total_speech) * 10.0
        continuity_score = max(0.0, min(100.0, continuity_score))

        LOG.info("=" * 60)
        LOG.info("Voice Continuity Report")
        LOG.info(f"  Total segments     : {len(segments)}")
        LOG.info(f"  Short segments (<4s): {short_segs}")
        LOG.info(f"  Ideal segments (8-12s): {ideal_segs}")
        LOG.info(f"  Profile switches   : 0 (Architecture enforced)")
        LOG.info(f"  Continuity Score   : {continuity_score:.1f}/100")
        LOG.info("=" * 60)

        manifest_path = self.out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

        try:
            del model
        except Exception:
            pass
        free_vram()
        log_vram(LOG)
        return {
            "generated_dir": str(self.out_dir),
            "manifest_path": str(manifest_path),
            "ref_transcripts": ref_transcripts,
        }
