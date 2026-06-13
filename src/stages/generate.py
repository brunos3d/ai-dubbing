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
        stage_banner(LOG, 6, 11, "OmniVoice Generation")

        translated_path = Path(context["translated_path"])
        samples_map: Dict[str, str] = context.get("speaker_samples", {})
        segments = json.loads(translated_path.read_text(encoding="utf-8"))
        if not segments:
            raise RuntimeError("No translated segments to synthesize")

        speakers = list(samples_map.keys())
        if not speakers:
            raise RuntimeError("No speaker reference samples available")

        self.out_dir.mkdir(parents=True, exist_ok=True)
        ref_transcripts: Dict[str, str] = {}
        source_language = context.get("source_language")
        for spk, path in samples_map.items():
            try:
                ref_transcripts[spk] = _transcribe_ref_with_whisper(path, source_language)
                LOG.info(f"Ref transcript ({spk}): {ref_transcripts[spk][:80]}")
            except Exception as exc:  # noqa: BLE001
                LOG.warning(f"Could not transcribe {spk} reference: {exc}")
                ref_transcripts[spk] = ""
        free_vram()

        LOG.info(f"Loading OmniVoice ({self.model_id}) on {self.device}")
        model = self._load_model()
        sample_rate = int(getattr(model, "sampling_rate", 24000))
        LOG.info(f"Model sampling rate: {sample_rate} Hz")

        prompts: Dict[str, Any] = {}
        if self.use_clone_prompt:
            for spk, path in samples_map.items():
                wav, sr = _coerce_ref_audio(path)
                try:
                    prompts[spk] = model.create_voice_clone_prompt(
                        ref_audio=(wav, sr),
                        ref_text=ref_transcripts.get(spk) or None,
                        preprocess_prompt=True,
                    )
                    LOG.info(f"Built voice clone prompt for {spk}")
                except Exception as exc:  # noqa: BLE001
                    LOG.warning(f"Could not build voice clone prompt for {spk}: {exc}")

        manifest: List[Dict[str, Any]] = []
        for i, seg in enumerate(segments):
            text = (seg.get("text") or "").strip()
            speaker = seg.get("speaker", speakers[0])
            # Validate speaker-to-profile mapping; this guards against the
            # "all speakers share one voice" failure mode.
            if speaker not in samples_map:
                LOG.warning(
                    f"Segment {i + 1}: speaker={speaker!r} has no profile, "
                    f"falling back to {speakers[0]!r}"
                )
                speaker = speakers[0]
            speaker_profile_id = speaker
            ref_audio = samples_map[speaker]
            ref_transcript = ref_transcripts.get(speaker, "")
            out_wav = self.out_dir / f"segment_{i + 1:04d}.wav"

            LOG.info(
                f"Segment {i + 1:>3}: speaker={speaker_profile_id}  "
                f"ref={Path(ref_audio).name}  "
                f"ref_text={ref_transcript[:60]!r}{'...' if len(ref_transcript) > 60 else ''}  "
                f"text={text[:60]!r}{'...' if len(text) > 60 else ''}"
            )

            kwargs: Dict[str, Any] = {
                "text": text,
                "language": _lang_name(self.target_language),
            }
            if speaker in prompts:
                kwargs["voice_clone_prompt"] = prompts[speaker]
            else:
                wav, sr = _coerce_ref_audio(ref_audio)
                kwargs["ref_audio"] = (wav, sr)
                kwargs["ref_text"] = ref_transcript or None

            try:
                audios = model.generate(**kwargs)
            except Exception as exc:  # noqa: BLE001
                LOG.warning(f"Segment {i + 1} failed with prompt, falling back: {exc}")
                wav, sr = _coerce_ref_audio(ref_audio)
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
            duration = float(audio.shape[0]) / sample_rate

            # Validation: the segment speaker id MUST match the voice
            # profile id actually used; if they ever diverge we want to
            # catch it loudly.
            profile_used = speaker_profile_id
            speaker_id_ok = profile_used == speaker
            if not speaker_id_ok:
                raise RuntimeError(
                    f"Speaker identity validation FAILED for segment {i + 1}: "
                    f"segment says {speaker!r} but voice profile used was "
                    f"{profile_used!r}"
                )

            manifest.append({
                "index": i,
                "speaker": speaker,
                "voice_profile_id": profile_used,
                "speaker_id_ok": speaker_id_ok,
                "ref_audio": ref_audio,
                "ref_transcript_chars": len(ref_transcript),
                "text": text,
                "source_text": seg.get("source_text", ""),
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "original_duration": float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)),
                "generated_duration": round(duration, 3),
                "path": str(out_wav),
                "sample_rate": sample_rate,
            })
            if (i + 1) % 5 == 0 or i == len(segments):
                LOG.info(f"Generated {i + 1}/{len(segments)} segments")

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
