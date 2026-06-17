"""Stage 5 - Speech recognition with faster-whisper."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.logging import get_logger, stage_banner
from ..utils.vram import free_vram, log_vram

LOG = get_logger("ai-dubbing.transcribe")


def clean_transcript_text(text: str) -> str:
    """Repair common ASR surface artifacts (Failure #4).

    Targets defect *classes* that are unambiguous from the text alone:

    * tokens carrying the Unicode replacement char ``U+FFFD`` (a decode
      failure mid-word, e.g. ``"Auf �ative"``) are dropped whole — the
      token is corrupt and there is nothing to salvage;
    * stutter runs of three or more identical consecutive words
      (``"initially initially initially"``) collapse to a single word;
    * stray control characters and runs of whitespace are normalised.

    Conservative by design: legitimate doubling (``"had had"``) and normal
    text pass through untouched, so this never invents or rewrites content.
    """
    if not text:
        return text

    # 1. Strip control chars (keep normal whitespace).
    text = "".join(
        ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 0x20
    )

    # 2. Drop tokens carrying a replacement char.  A decode failure mid-word
    #    makes Whisper split the broken word around the char ("Auf �ative"),
    #    so when the char sits at a token edge we also drop the glued fragment
    #    on that side (it is the other half of the same corrupt word).
    if "�" in text:
        tokens = text.split(" ")
        drop = [False] * len(tokens)
        for i, t in enumerate(tokens):
            if "�" not in t:
                continue
            drop[i] = True
            if t.startswith("�") and i > 0:          # tail fragment -> head before it
                drop[i - 1] = True
            if t.endswith("�") and i + 1 < len(tokens):  # head fragment -> tail after it
                drop[i + 1] = True
        text = " ".join(t for i, t in enumerate(tokens) if not drop[i])

    # 3. Collapse stutter runs (>= 3 identical consecutive words, case-insensitive).
    text = " ".join(_collapse_stutters(text.split()))

    # 4. Normalise whitespace.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _collapse_stutters(words: List[str]) -> List[str]:
    """Collapse runs of >= 3 identical consecutive words to one; keep <= 2."""
    out: List[str] = []
    i = 0
    n = len(words)
    while i < n:
        j = i
        while j + 1 < n and words[j + 1].lower() == words[i].lower():
            j += 1
        run = j - i + 1
        keep = 1 if run >= 3 else run
        out.extend(words[i : i + keep])
        i = j + 1
    return out


def _reverify_low_confidence(
    speech_path: Path,
    segment: Dict[str, Any],
    model_size: str,
    language: Optional[str],
    device: str,
    compute_type: str,
    threshold: float = 0.55,
    model: Optional[Any] = None,
) -> Dict[str, Any]:
    """Re-transcribe a low-confidence segment with higher effort (larger beam).

    ``avg_logprob`` on our segments is a mean *word probability* in [0, 1]
    (see :func:`_finalize_segment`), not a log-prob, so the gate compares
    against a probability threshold.  A segment at or above ``threshold`` is
    confident enough and skipped; below it we re-run Whisper with a larger
    beam and keep the result only if it is more probable.
    """
    if segment.get("avg_logprob", 0.0) >= threshold:
        return segment

    from faster_whisper import WhisperModel

    LOG.info(f"Re-verifying low-confidence segment ({segment['avg_logprob']:.2f}): {segment['text'][:50]}...")
    
    local_model = False
    if model is None:
        try:
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            local_model = True
        except Exception as exc:
            LOG.warning(f"Failed to load re-verification model: {exc}")
            return segment

    try:
        lang = None if (not language or language == "auto") else language
        
        # Crop audio to just this segment for focused re-verification
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
        if local_model:
            try:
                del model
            except Exception:
                pass
            free_vram()
        
    return segment


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
    text = clean_transcript_text(" ".join(w["word"] for w in words).strip())
    
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


def _assign_word_speaker(
    w_start: float, w_end: float, segments: List[Dict[str, Any]]
) -> str:
    """Assign a single word to a diarization speaker.

    A word is short, so we first try the diarization turn that *overlaps*
    it (max overlap wins).  Words that fall in a silent gap between turns
    get the *nearest* turn by midpoint distance instead of the sentinel
    ``speaker_unknown`` - this avoids fragmenting an utterance every time
    Whisper's word boundary lands a few milliseconds outside pyannote's.
    """
    if not segments:
        return "speaker_unknown"
    best_speaker = ""
    best_overlap = 0.0
    for s in segments:
        overlap = max(0.0, min(w_end, s["end"]) - max(w_start, s["start"]))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = s["speaker"]
    if best_speaker:
        return best_speaker
    # No overlap: snap to the nearest turn by distance from the word midpoint.
    mid = 0.5 * (w_start + w_end)
    nearest = min(
        segments,
        key=lambda s: 0.0
        if s["start"] <= mid <= s["end"]
        else min(abs(mid - s["start"]), abs(mid - s["end"])),
    )
    return nearest["speaker"]


def _segment_words_by_speaker(
    words: List[Dict[str, Any]],
    diar_segments: List[Dict[str, Any]],
    max_gap: float = 0.8,
    target_duration: float = 10.0,
    max_duration: float = 15.0,
) -> List[Dict[str, Any]]:
    """Tag every word with a speaker, then group into utterances.

    Unlike the old per-Whisper-segment assignment, this splits the word
    stream wherever the *word-level* speaker changes, so a Whisper block
    that straddles a turn boundary becomes two correctly-attributed
    utterances instead of one mis-attributed one (Failure #1).
    """
    tagged = sorted(words, key=lambda w: w["start"])
    for w in tagged:
        w["speaker"] = _assign_word_speaker(w["start"], w["end"], diar_segments)

    out: List[Dict[str, Any]] = []
    run: List[Dict[str, Any]] = []
    cur: Optional[str] = None
    for w in tagged:
        if cur is None or w["speaker"] == cur:
            run.append(w)
            cur = w["speaker"]
        else:
            out.extend(
                _group_words_into_segments(run, cur, max_gap, target_duration, max_duration)
            )
            run = [w]
            cur = w["speaker"]
    if run and cur is not None:
        out.extend(
            _group_words_into_segments(run, cur, max_gap, target_duration, max_duration)
        )
    out.sort(key=lambda s: s["start"])
    return out


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
    inputs: List[str] = ["media/speech.wav", "diarization/segments.json"]
    outputs: List[str] = [
        "transcription/transcript.json",
        "transcription/word_level.json",
    ]
    editable_outputs: List[str] = ["transcription/transcript.json"]
    derived_outputs: List[str] = []
    config_fields: List[str] = ["model_size", "source_language", "device"]

    def __init__(
        self,
        workdir: Path,
        model_size: str = "large-v3",
        source_language: Optional[str] = None,
        device: str = "cuda",
        subdir: str | None = None,
    ):
        self.workdir = Path(workdir)
        if subdir:
            self.workdir = self.workdir / subdir
        self.model_size = model_size
        self.source_language = source_language
        self.device = device
        self.subdir = subdir

    def output_paths(self) -> List[Path]:
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

        # Word-level speaker attribution (Failure #1): assign a speaker to
        # every word from diarization and split utterances at turn changes,
        # rather than stamping one speaker onto a whole Whisper segment that
        # may span a turn boundary.
        all_words = [ww for seg in word_segments for ww in seg["words"]]
        merged = _segment_words_by_speaker(all_words, diar_segments)
        seg_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))

        # --- Second Pass / Refinement ---
        LOG.info("Refining transcript (low-confidence re-verification)...")

        # 1. Re-verify low-confidence segments.
        # We pre-load the model once for all re-verifications to avoid reload overhead.
        from faster_whisper import WhisperModel
        try:
            rev_model = WhisperModel(self.model_size, device=self.device, compute_type=compute_type)
        except Exception as exc:
            LOG.warning(f"Could not load re-verification model: {exc}")
            rev_model = None

        refined: List[Dict[str, Any]] = []
        for s in merged:
            rs = _reverify_low_confidence(
                speech_path, s, self.model_size,
                self.source_language, self.device, compute_type,
                model=rev_model,
            )
            refined.append(rs)

        if rev_model:
            del rev_model
            free_vram()

        seg_path.write_text(json.dumps(refined, indent=2, ensure_ascii=False))
        LOG.info(f"Transcript: {len(refined)} utterances")
        log_vram(LOG)
        return {
            "transcript_path": str(seg_path),
            "word_path": str(word_path),
            "num_utterances": len(refined),
        }
