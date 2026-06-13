"""Stage 4 - Build per-speaker voice profiles for voice cloning.

For every diarized speaker we produce a structured profile:

    working/speaker_profiles/speaker_NN/
    ├── reference.wav     # 5-10s concatenated clean speech, mono
    ├── transcript.txt     # Whisper transcript of reference.wav
    └── metadata.json     # duration, num regions used, word count, etc.

We *also* keep a flat copy at ``working/speakers/speaker_NN.wav`` so the
existing generate / align stages can look the speaker up by name without
knowing about the per-speaker directory structure.

The reference is built by:

1. Collecting all diarized segments for the speaker.
2. Splitting each segment into "speech" vs "silence" chunks using a
   short-window RMS energy threshold (so intro music and trailing silence
   never leak into the reference).
3. Scoring every speech chunk (RMS * length) and greedily picking the best
   ones, concatenating them with small (50 ms) gaps until we hit the target
   duration (5 s) — capped at 10 s as required by OmniVoice.
4. Crossfading the joints so the resulting reference sounds like a single
   continuous utterance, not a stitched-together collage.
5. Transcribing the final reference with faster-whisper so OmniVoice gets
   the text alongside the audio (avoids the "speaker counts to 10 then
   keeps repeating numbers" failure mode).
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.audio import compute_snr_db, read_wav, rms_normalize, write_wav
from ..utils.logging import get_logger, stage_banner
from ..utils.vram import free_vram, log_vram

LOG = get_logger("ai-dubbing.samples")


# ---------------------------------------------------------------------------
# Chunk extraction
# ---------------------------------------------------------------------------

@dataclass
class SpeechChunk:
    """A voiced region inside a diarized segment."""
    speaker: str
    start_sample: int        # absolute position in the speech.wav array
    end_sample: int
    score: float             # higher = better candidate for the reference

    @property
    def duration_s(self) -> float:
        return 0.0  # populated by caller once the sample rate is known


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 2:
        if audio.shape[0] > 1:
            return audio.mean(axis=0)
        return audio[0]
    return audio


def _split_into_speech_chunks(
    mono: np.ndarray,
    sr: int,
    seg_start_sample: int,
    seg_end_sample: int,
    *,
    frame_ms: int = 20,
    speech_db: float = -40.0,
    min_chunk_ms: int = 200,
    pad_ms: int = 60,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Find voiced regions inside the given segment.

    Returns ``(chunks, frame_energies)`` where each chunk is an
    ``(start_sample, end_sample)`` absolute index into ``mono``.
    """
    n = seg_end_sample - seg_start_sample
    if n <= 0:
        return [], np.zeros(0)
    seg = mono[seg_start_sample:seg_end_sample]
    frame_len = max(1, int(sr * frame_ms / 1000))
    hop = frame_len
    n_frames = max(1, n // hop)
    pad = max(0, n - n_frames * hop)
    if pad > 0:
        seg = seg[:-pad]
        n = seg.shape[0]
    frames = np.lib.stride_tricks.as_strided(
        seg,
        shape=(n_frames, frame_len),
        strides=(seg.strides[0] * hop, seg.strides[0]),
    )
    energies = np.sqrt(np.mean(frames.astype("float32") ** 2, axis=1) + 1e-12)
    # Adaptive threshold: max of an absolute floor and 20% of the segment's
    # loudest frame.  This handles demucs vocals which can be very quiet
    # but still well above the silence floor of the same clip.
    seg_peak = float(np.max(energies)) if energies.size else 0.0
    threshold = max(10 ** (speech_db / 20.0), seg_peak * 0.20)
    is_speech = energies > threshold

    min_chunk_frames = max(1, int(min_chunk_ms / frame_ms))
    pad_frames = max(0, int(pad_ms / frame_ms))

    chunks: List[Tuple[int, int]] = []
    i = 0
    while i < n_frames:
        if not is_speech[i]:
            i += 1
            continue
        j = i
        while j < n_frames and is_speech[j]:
            j += 1
        if j - i >= min_chunk_frames:
            s_frame = max(0, i - pad_frames)
            e_frame = min(n_frames, j + pad_frames)
            s = seg_start_sample + s_frame * hop
            e = seg_start_sample + e_frame * hop
            chunks.append((s, e))
        i = j
    return chunks, energies


def _collect_speaker_chunks(
    mono: np.ndarray,
    sr: int,
    diarized_segments: List[Dict[str, Any]],
    speaker: str,
) -> List[SpeechChunk]:
    """All voiced chunks for one speaker, scored for quality."""
    out: List[SpeechChunk] = []
    for dseg in diarized_segments:
        if dseg.get("speaker") != speaker:
            continue
        s_abs = max(0, int(dseg["start"] * sr))
        e_abs = min(mono.shape[-1], int(dseg["end"] * sr))
        if e_abs - s_abs < int(0.05 * sr):
            continue
        chunks, _ = _split_into_speech_chunks(mono, sr, s_abs, e_abs)
        for cs, ce in chunks:
            clip = mono[cs:ce]
            if clip.size < int(0.1 * sr):
                continue
            rms = float(np.sqrt(np.mean(clip**2)))
            peak = float(np.max(np.abs(clip)))
            length = (ce - cs) / sr
            if peak < 1e-4 or rms < 1e-4:
                continue
            score = rms * length * 1000.0  # weighted by length so a long
                                            # loud chunk beats a short one
            out.append(SpeechChunk(speaker, cs, ce, score))
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def _concatenate_chunks(
    mono: np.ndarray,
    sr: int,
    chunks: List[SpeechChunk],
    target_seconds: float = 5.0,
    max_seconds: float = 10.0,
    gap_ms: int = 60,
) -> Tuple[np.ndarray, List[SpeechChunk]]:
    """Greedily concatenate the best chunks until we reach the target.

    Returns ``(reference_audio, used_chunks)``. The audio is mono, with
    small silence gaps between joined chunks (no crossfade, since OmniVoice
    expects a clean sample — small gaps read as natural pauses to the model).
    """
    if not chunks:
        return np.zeros(0, dtype="float32"), []
    target = max(0.5, target_seconds)
    cap = max(target, max_seconds)
    gap = int(gap_ms / 1000 * sr)
    out_segments: List[np.ndarray] = []
    used: List[SpeechChunk] = []
    total = 0.0
    for ch in chunks:
        clip = mono[ch.start_sample:ch.end_sample]
        dur = clip.shape[0] / sr
        if dur > cap - total:
            # Trim to fit
            keep_samples = int((cap - total) * sr)
            if keep_samples < int(0.1 * sr):
                break
            clip = clip[:keep_samples]
            dur = clip.shape[0] / sr
        if out_segments:
            out_segments.append(np.zeros(gap, dtype="float32"))
            total += gap / sr
        out_segments.append(clip)
        used.append(ch)
        total += dur
        if total >= target:
            break
    if not out_segments:
        return np.zeros(0, dtype="float32"), []
    ref = np.concatenate(out_segments).astype("float32")
    if ref.shape[0] / sr > cap:
        ref = ref[: int(cap * sr)]
    return ref, used


# ---------------------------------------------------------------------------
# Whisper transcript for the reference
# ---------------------------------------------------------------------------

def _transcribe_reference(audio_path: Path, source_language: Optional[str]) -> str:
    """Run faster-whisper (tiny) on a short reference clip.

    Returns the joined transcript text. Empty string on failure.
    """
    try:
        from faster_whisper import WhisperModel

        LOG.info(f"Transcribing reference {audio_path.name} with faster-whisper (tiny)")
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        try:
            lang = source_language if source_language and source_language != "auto" else None
            segs, _info = model.transcribe(
                str(audio_path),
                language=lang,
                beam_size=1,
                vad_filter=True,
            )
            text = " ".join(s.text.strip() for s in segs).strip()
        finally:
            try:
                del model
            except Exception:
                pass
            free_vram()
        return text
    except Exception as exc:  # noqa: BLE001
        LOG.warning(f"Whisper transcription of {audio_path} failed: {exc}")
        return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_speaker_profiles(
    speech_path: Path,
    diarized_segments: List[Dict[str, Any]],
    profiles_dir: Path,
    *,
    source_language: Optional[str] = None,
    target_seconds: float = 7.0,
    max_seconds: float = 10.0,
    flat_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build one voice profile per speaker.

    Returns a mapping ``speaker_id -> {"reference": path, "transcript": path,
    "metadata": path, "duration_s": float, "word_count": int}``.
    """
    profiles_dir.mkdir(parents=True, exist_ok=True)
    if flat_dir is not None:
        flat_dir.mkdir(parents=True, exist_ok=True)

    audio, sr = read_wav(speech_path)
    mono = _to_mono(audio)
    speakers = sorted({s["speaker"] for s in diarized_segments})

    out: Dict[str, Dict[str, Any]] = {}
    for spk in speakers:
        spk_dir = profiles_dir / spk
        spk_dir.mkdir(parents=True, exist_ok=True)
        chunks = _collect_speaker_chunks(mono, sr, diarized_segments, spk)
        if not chunks:
            LOG.warning(f"  {spk}: no voiced chunks found, skipping profile")
            continue
        ref_audio, used = _concatenate_chunks(
            mono, sr, chunks,
            target_seconds=target_seconds,
            max_seconds=max_seconds,
        )
        if ref_audio.size < int(0.5 * sr):
            LOG.warning(
                f"  {spk}: only {ref_audio.shape[0]/sr:.2f}s of speech; "
                f"profile may be poor"
            )
        ref_audio = rms_normalize(ref_audio, target_dbfs=-20.0)
        ref_path = spk_dir / "reference.wav"
        write_wav(ref_path, ref_audio, sr)

        transcript = _transcribe_reference(ref_path, source_language)
        transcript_path = spk_dir / "transcript.txt"
        transcript_path.write_text(transcript + "\n", encoding="utf-8")
        word_count = len(transcript.split())

        duration_s = ref_audio.shape[0] / sr
        metadata = {
            "speaker_id": spk,
            "reference_duration": round(duration_s, 3),
            "reference_sample_rate": int(sr),
            "segments_used": len({id(c) for c in used}),
            "transcript_words": word_count,
            "transcript_chars": len(transcript),
            "reference_path": str(ref_path),
            "transcript_path": str(transcript_path),
        }
        meta_path = spk_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

        # Flat alias so generate / align can look up speaker -> wav.
        if flat_dir is not None:
            flat = flat_dir / f"{spk}.wav"
            shutil.copy2(ref_path, flat)

        out[spk] = {
            "reference": str(ref_path),
            "transcript": str(transcript_path),
            "metadata": str(meta_path),
            "duration_s": duration_s,
            "word_count": word_count,
            "transcript_text": transcript,
        }
        LOG.info(
            f"  {spk}: {duration_s:5.2f}s ref, {len(used)} chunk(s) joined, "
            f"{word_count} words transcript"
        )
    return out


# ---------------------------------------------------------------------------
# Stage wrapper
# ---------------------------------------------------------------------------

class SampleStage:
    name = "samples"

    def __init__(
        self,
        workdir: Path,
        target_seconds: float = 7.0,
        max_seconds: float = 10.0,
    ):
        self.workdir = Path(workdir)
        self.target_seconds = target_seconds
        self.max_seconds = max_seconds

    def outputs(self) -> List[Path]:
        return [self.workdir / "speaker_profiles", self.workdir / "speakers"]

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        stage_banner(LOG, 3, 11, "Sample Extraction")
        speech_path = Path(context["speech_path"])
        segments_path = Path(context["segments_path"])
        source_language = context.get("source_language")
        profiles_dir = self.workdir / "speaker_profiles"
        flat_dir = self.workdir / "speakers"

        diarized_segments = json.loads(segments_path.read_text(encoding="utf-8"))
        profiles = build_speaker_profiles(
            speech_path,
            diarized_segments,
            profiles_dir,
            source_language=source_language,
            target_seconds=self.target_seconds,
            max_seconds=self.max_seconds,
            flat_dir=flat_dir,
        )

        # Backwards-compatible mapping for the generate / align stages.
        speaker_samples = {spk: p["reference"] for spk, p in profiles.items()}
        ref_transcripts = {spk: p["transcript_text"] for spk, p in profiles.items()}

        # Final diagnostic block — answers "how many speakers, and how much
        # voice material did we extract for each?"
        LOG.info("=" * 60)
        LOG.info(f"Voice profiles built: {len(profiles)}")
        for spk in sorted(profiles):
            p = profiles[spk]
            quality = "ok" if p["duration_s"] >= 5 and p["word_count"] >= 5 else "POOR"
            LOG.info(
                f"  {spk}: duration={p['duration_s']:.2f}s  "
                f"transcript_words={p['word_count']}  ref={p['reference']}  [{quality}]"
            )
            transcript = p["transcript_text"]
            if transcript:
                LOG.info(f"    transcript: {transcript[:120]}{'...' if len(transcript) > 120 else ''}")
        LOG.info("=" * 60)

        log_vram(LOG)
        free_vram()
        return {
            "speaker_samples": speaker_samples,
            "ref_transcripts": ref_transcripts,
            "speaker_profiles": profiles,
            "profiles_dir": str(profiles_dir),
            "speakers_dir": str(flat_dir),
        }
