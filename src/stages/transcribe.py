"""Stage 5 - Speech recognition with faster-whisper."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.logging import get_logger, stage_banner
from ..utils.vram import free_vram, log_vram

LOG = get_logger("ai-dubbing.transcribe")


def _default_compute_type(device: str) -> str:
    if device.startswith("cuda"):
        try:
            import torch

            free, _ = torch.cuda.mem_get_info()
            free_gb = free / (1024 ** 3)
            if free_gb < 5.0:
                return "int8"
        except Exception:
            pass
        return "float16"
    return "int8"


def _check_vram_or_skip(needed_gb: float = 1.5) -> bool:
    """Return True if there is enough free VRAM for inference, False otherwise."""
    try:
        import torch

        free, _ = torch.cuda.mem_get_info()
        return (free / (1024 ** 3)) >= needed_gb
    except Exception:
        return False


def _group_words_into_segments(
    words: List[Dict[str, Any]],
    speaker: str,
    max_gap: float = 0.6,
) -> List[Dict[str, Any]]:
    """Group word-level timestamps into utterance-level segments per speaker."""
    out: List[Dict[str, Any]] = []
    buf: List[Dict[str, Any]] = []
    for w in words:
        if buf and (w["start"] - buf[-1]["end"]) > max_gap:
            out.append(_finalize_segment(buf, speaker))
            buf = []
        buf.append(w)
    if buf:
        out.append(_finalize_segment(buf, speaker))
    return [s for s in out if s["text"].strip()]


def _finalize_segment(words: List[Dict[str, Any]], speaker: str) -> Dict[str, Any]:
    return {
        "speaker": speaker,
        "start": round(words[0]["start"], 3),
        "end": round(words[-1]["end"], 3),
        "text": " ".join(w["word"] for w in words).strip(),
        "avg_logprob": sum(w.get("logprob", 0.0) for w in words) / max(1, len(words)),
    }


def _assign_speaker(t_start: float, t_end: float, segments: List[Dict[str, Any]]) -> str:
    best_speaker = "speaker_unknown"
    best_overlap = 0.0
    for s in segments:
        overlap = max(0.0, min(t_end, s["end"]) - max(t_start, s["start"]))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = s["speaker"]
    return best_speaker


def transcribe_audio(
    speech_path: Path,
    segments: List[Dict[str, Any]],
    model_size: str,
    language: Optional[str],
    device: str,
    compute_type: str,
    beam_size: int = 3,
) -> List[Dict[str, Any]]:
    from faster_whisper import WhisperModel

    LOG.info(f"Loading faster-whisper {model_size} on {device} ({compute_type})")
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and device.startswith("cuda"):
            LOG.warning(f"OOM loading model on {device} ({compute_type}); retrying on CPU")
            free_vram()
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
        else:
            raise
    try:
        lang = None if (not language or language == "auto") else language
        try:
            segments_iter, info = model.transcribe(
                str(speech_path),
                language=lang,
                beam_size=beam_size,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                word_timestamps=True,
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and device.startswith("cuda"):
                LOG.warning("OOM during transcribe; retrying on CPU with int8")
                free_vram()
                del model
                model = WhisperModel(model_size, device="cpu", compute_type="int8")
                segments_iter, info = model.transcribe(
                    str(speech_path),
                    language=lang,
                    beam_size=1,
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 500},
                    word_timestamps=True,
                )
            else:
                raise
        all_segments: List[Dict[str, Any]] = []
        for seg in segments_iter:
            seg_dict = {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip(),
                "words": [
                    {
                        "word": w.word.strip(),
                        "start": float(w.start),
                        "end": float(w.end),
                        "logprob": float(getattr(w, "probability", 0.0) or 0.0),
                    }
                    for w in (seg.words or [])
                ],
            }
            seg_dict["speaker"] = _assign_speaker(seg_dict["start"], seg_dict["end"], segments)
            all_segments.append(seg_dict)
        LOG.info(f"Detected language: {info.language} (prob {info.language_probability:.2f})")
        return all_segments
    finally:
        try:
            del model
        except Exception:
            pass
        free_vram()


class TranscribeStage:
    name = "transcribe"

    def __init__(
        self,
        workdir: Path,
        model_size: str = "large-v3",
        source_language: Optional[str] = None,
        device: str = "cuda",
    ):
        self.workdir = Path(workdir)
        self.model_size = model_size
        self.source_language = source_language
        self.device = device

    def outputs(self) -> List[Path]:
        return [self.workdir / "transcript.json", self.workdir / "transcript_word_level.json"]

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        stage_banner(LOG, 4, 11, "Whisper Transcription")
        speech_path = Path(context["speech_path"])
        segments_path = Path(context["segments_path"])
        diar_segments = json.loads(segments_path.read_text(encoding="utf-8"))
        word_path = self.workdir / "transcript_word_level.json"
        seg_path = self.workdir / "transcript.json"

        compute_type = _default_compute_type(self.device)
        if self.device.startswith("cuda") and not _check_vram_or_skip(2.0):
            LOG.warning("Not enough free VRAM; running Whisper on CPU")
            self.device = "cpu"
            compute_type = "int8"
        try:
            word_segments = transcribe_audio(
                speech_path, diar_segments, self.model_size,
                self.source_language, self.device, compute_type,
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and self.device.startswith("cuda"):
                LOG.warning(f"Whisper OOM on {self.device}; downgrading to CPU int8")
                free_vram()
                word_segments = transcribe_audio(
                    speech_path, diar_segments, self.model_size,
                    self.source_language, "cpu", "int8",
                )
            else:
                raise
        word_path.write_text(json.dumps(word_segments, indent=2, ensure_ascii=False))

        merged: List[Dict[str, Any]] = []
        for spk in sorted({s["speaker"] for s in diar_segments}):
            spk_words = [w for w in word_segments if w["speaker"] == spk]
            words = [ww for seg in spk_words for ww in seg["words"]]
            merged.extend(_group_words_into_segments(words, spk))
        merged.sort(key=lambda s: s["start"])
        seg_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
        LOG.info(f"Transcript: {len(merged)} utterances")
        log_vram(LOG)
        return {
            "transcript_path": str(seg_path),
            "word_path": str(word_path),
            "num_utterances": len(merged),
        }
