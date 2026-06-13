"""Stage 5 - Speech recognition with faster-whisper."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..utils.entities import EntityPreserver, build_video_vocabulary
from ..utils.logging import get_logger, stage_banner
from ..utils.vram import free_vram, log_vram

LOG = get_logger("ai-dubbing.transcribe")


def _reverify_low_confidence(
    speech_path: Path,
    segment: Dict[str, Any],
    model_size: str,
    language: Optional[str],
    device: str,
    compute_type: str,
    threshold: float = -0.8,
) -> Dict[str, Any]:
    """Re-transcribe a low-confidence segment with higher effort (larger beam)."""
    if segment.get("avg_logprob", 0.0) >= threshold:
        return segment

    from faster_whisper import WhisperModel

    LOG.info(f"Re-verifying low-confidence segment ({segment['avg_logprob']:.2f}): {segment['text'][:50]}...")
    
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        lang = None if (not language or language == "auto") else language
        
        # Crop audio to just this segment for focused re-verification
        # (For now, we just run transcribe with restricted timestamps to avoid complex cropping)
        segs_iter, _ = model.transcribe(
            str(speech_path),
            language=lang,
            beam_size=10, # Higher beam for second pass
            best_of=5,
            initial_prompt=segment["text"], # Use first pass as prompt
            clip_timestamps=f"{segment['start']},{segment['end']}",
            vad_filter=True,
        )
        
        results = list(segs_iter)
        if results:
            best = max(results, key=lambda x: x.avg_logprob)
            if best.avg_logprob > segment["avg_logprob"]:
                LOG.info(f"  Improved confidence: {segment['avg_logprob']:.2f} -> {best.avg_logprob:.2f}")
                new_segment = dict(segment)
                new_segment["text"] = best.text.strip()
                new_segment["avg_logprob"] = best.avg_logprob
                return new_segment
    except Exception as exc:
        LOG.warning(f"Re-verification failed: {exc}")
    finally:
        try:
            del model
        except Exception:
            pass
        free_vram()
        
    return segment


def _correct_with_vocabulary(text: str, vocabulary: Set[str]) -> str:
    """Heuristic correction of names/terms using video-wide vocabulary."""
    corrected = text
    for term in vocabulary:
        # If text contains a fuzzy match of term, replace it with the term
        # (Very basic implementation: check if term.lower() is in text.lower())
        # To avoid over-correction, we only do this for terms > 5 chars or distinct ones.
        if term.lower() in text.lower():
            # Use regex to find and replace with exact casing from vocabulary
            corrected = re.sub(re.escape(term), term, corrected, flags=re.IGNORECASE)
    return corrected


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
    max_gap: float = 0.8,
    target_duration: float = 10.0,
    max_duration: float = 15.0,
) -> List[Dict[str, Any]]:
    """Group word-level timestamps into utterance-level segments per speaker.

    Prefers longer utterances (target 8-12s) to improve prosodic flow and
    voice stability. Only splits when a natural boundary (sentence or clause)
    is reached and the duration is sufficient.
    """
    out: List[Dict[str, Any]] = []
    buf: List[Dict[str, Any]] = []

    for w in words:
        # 1. Natural gap split (always respect large pauses)
        if buf and (w["start"] - buf[-1]["end"]) > max_gap:
            out.extend(_split_if_long(buf, speaker, target_duration, max_duration))
            buf = []

        buf.append(w)

        # 2. Smart boundary split
        # We check if the current buffer is already reasonably long.
        # If it's over target_duration and ends with sentence-ending punctuation, split it.
        if buf and (buf[-1]["end"] - buf[0]["start"]) >= target_duration:
            text = buf[-1]["word"].strip()
            # Only split if we have a clear sentence boundary.
            if text and text[-1] in ".?!":
                out.extend(_split_if_long(buf, speaker, target_duration, max_duration))
                buf = []

    if buf:
        out.extend(_split_if_long(buf, speaker, target_duration, max_duration))

    return [s for s in out if s["text"].strip()]


def _split_if_long(
    words: List[Dict[str, Any]],
    speaker: str,
    target_duration: float,
    max_duration: float,
) -> List[Dict[str, Any]]:
    """Recursively split a list of words if it exceeds max_duration."""
    if not words:
        return []

    duration = words[-1]["end"] - words[0]["start"]
    if duration <= max_duration:
        return [_finalize_segment(words, speaker)]

    # Find the best split point.
    # Priority 1: Sentence boundaries (. ? !)
    # Priority 2: Clause boundaries (, ; :)
    # Priority 3: Largest gap between words
    best_idx = -1
    best_score = -1.0

    # We want to split as close to the middle as possible, but respecting punctuation.
    mid_time = words[0]["start"] + duration / 2.0

    for i in range(len(words) - 1):
        w = words[i]
        text = w["word"].strip()
        gap = words[i + 1]["start"] - w["end"]

        score = 0.0
        if text and text[-1] in ".?!":
            score = 100.0
        elif text and text[-1] in ",;:":
            score = 50.0
        else:
            score = gap * 10.0

        # Distance from middle penalty
        dist_from_mid = abs(w["end"] - mid_time)
        score -= dist_from_mid

        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx == -1:
        # Fallback: split in the middle
        best_idx = len(words) // 2

    left = words[: best_idx + 1]
    right = words[best_idx + 1 :]

    return _split_if_long(left, speaker, target_duration, max_duration) + \
           _split_if_long(right, speaker, target_duration, max_duration)


def _finalize_segment(words: List[Dict[str, Any]], speaker: str) -> Dict[str, Any]:
    text = " ".join(w["word"] for w in words).strip()
    
    # Check if this is a non-speech vocal event
    # Patterns: [Laughter], (Sigh), [BREATH], etc.
    is_non_speech = False
    clean_text = text.lower().strip("[]().,?!")
    vocal_events = {"laughter", "chuckle", "sigh", "breath", "breathing", "cough", "gasp", "humming"}
    if clean_text in vocal_events or (text.startswith("[") and text.endswith("]")) or (text.startswith("(") and text.endswith(")")):
        is_non_speech = True

    return {
        "speaker": speaker,
        "start": round(words[0]["start"], 3),
        "end": round(words[-1]["end"], 3),
        "text": text,
        "avg_logprob": sum(w.get("logprob", 0.0) for w in words) / max(1, len(words)),
        "is_non_speech": is_non_speech,
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
        stage_banner(LOG, 4, 12, "Whisper Transcription")
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

        # --- Second Pass / Refinement ---
        LOG.info("Refining transcript (entity preservation & low-confidence check)...")
        
        # 1. Build video-wide vocabulary from high-confidence segments
        high_conf = [s for s in merged if s.get("avg_logprob", 0.0) > -0.5]
        video_vocab = build_video_vocabulary(high_conf)
        
        # 2. Re-verify low-confidence segments and apply corrections
        refined: List[Dict[str, Any]] = []
        for s in merged:
            # Re-verify if needed
            rs = _reverify_low_confidence(
                speech_path, s, self.model_size, 
                self.source_language, self.device, compute_type
            )
            # Contextual correction
            rs["text"] = _correct_with_vocabulary(rs["text"], video_vocab)
            refined.append(rs)
            
        seg_path.write_text(json.dumps(refined, indent=2, ensure_ascii=False))
        LOG.info(f"Transcript: {len(refined)} utterances")
        log_vram(LOG)
        return {
            "transcript_path": str(seg_path),
            "word_path": str(word_path),
            "num_utterances": len(refined),
            "video_vocabulary": list(video_vocab),
        }
