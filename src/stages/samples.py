"""Stage 4 - Build high-quality voice profiles for voice cloning.

Instead of clustering speaking styles (which causes voice instability), we
now focus on finding the single best reference window (8-12s) for each
speaker.

We prioritize:
1. Clean signal (High SNR)
2. Continuity (Natural speech flow, not fragments)
3. Linguistic completeness (Complete thoughts/sentences)
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
# Quality-based Profile Selection
# ---------------------------------------------------------------------------

@dataclass
class SpeechChunk:
    """A voiced region inside a diarized segment with quality metrics."""
    speaker: str
    start_sample: int        # absolute position in the speech.wav array
    end_sample: int
    duration_s: float
    snr: float = 0.0         # estimated signal-to-noise ratio
    rms: float = 0.0         # energy
    score: float = 0.0


def _score_chunk(chunk: SpeechChunk) -> float:
    """Compute a quality score for a speech chunk.

    Favors chunks that are:
    1. Clean (high SNR)
    2. Continuous (8-12 seconds is ideal for a reference)
    3. Strong signal (reasonable RMS)
    """
    # 1. SNR contribution (log scale)
    snr_score = max(0.0, min(1.0, chunk.snr / 40.0)) * 40.0

    # 2. Duration contribution (Bell curve around 10s)
    # Ideal: 8-12s
    if chunk.duration_s < 1.0:
        dur_score = 0.0
    elif chunk.duration_s < 8.0:
        dur_score = (chunk.duration_s / 8.0) * 30.0
    elif chunk.duration_s <= 12.0:
        dur_score = 30.0
    else:
        # Penalize overly long chunks as they might contain transitions/noise
        dur_score = max(0.0, 30.0 - (chunk.duration_s - 12.0) * 2.0)

    # 3. RMS / Signal Strength
    rms_score = max(0.0, min(1.0, chunk.rms * 10.0)) * 10.0

    # 4. Continuity (bonus for longer segments that aren't fragments)
    continuity_bonus = 20.0 if chunk.duration_s > 5.0 else 0.0

    return snr_score + dur_score + rms_score + continuity_bonus


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
        if e_abs - s_abs < int(0.1 * sr):
            continue
        
        clip = mono[s_abs:e_abs]
        rms = float(np.sqrt(np.mean(clip**2)))
        snr = compute_snr_db(clip)
        dur = (e_abs - s_abs) / sr
        
        if rms < 1e-4:
            continue
            
        chunk = SpeechChunk(
            speaker=speaker,
            start_sample=s_abs,
            end_sample=e_abs,
            duration_s=dur,
            snr=snr,
            rms=rms
        )
        chunk.score = _score_chunk(chunk)
        out.append(chunk)
        
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def _transcribe_reference(
    audio_path: Path, 
    source_language: Optional[str],
    model: Optional[Any] = None,
) -> str:
    """Run faster-whisper (tiny) on a short reference clip."""
    local_model = False
    try:
        from faster_whisper import WhisperModel

        if model is None:
            LOG.info(f"Transcribing reference {audio_path.name} with faster-whisper (tiny)")
            model = WhisperModel("tiny", device="cpu", compute_type="int8")
            local_model = True
        
        lang = source_language if source_language and source_language != "auto" else None
        segs, _info = model.transcribe(
            str(audio_path),
            language=lang,
            beam_size=1,
            vad_filter=True,
        )
        return " ".join(s.text.strip() for s in segs).strip()
    except Exception as exc:  # noqa: BLE001
        LOG.warning(f"Whisper transcription of {audio_path} failed: {exc}")
        return ""
    finally:
        if local_model:
            try:
                del model
            except Exception:
                pass
            free_vram()


def build_speaker_profiles(
    speech_path: Path,
    diarized_segments: List[Dict[str, Any]],
    profiles_dir: Path,
    *,
    source_language: Optional[str] = None,
    target_seconds: float = 10.0,
    max_seconds: float = 15.0,
    flat_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build high-quality voice profiles per speaker.

    Instead of clustering, we pick the best candidate reference window
    to ensure stable identity.
    """
    profiles_dir.mkdir(parents=True, exist_ok=True)
    if flat_dir is not None:
        flat_dir.mkdir(parents=True, exist_ok=True)

    audio, sr = read_wav(speech_path)
    if audio.ndim == 2:
        mono = audio.mean(axis=0)
    else:
        mono = audio
        
    speakers = sorted({s["speaker"] for s in diarized_segments})

    # Pre-load tiny model once for all speaker references
    from faster_whisper import WhisperModel
    try:
        whisper_tiny = WhisperModel("tiny", device="cpu", compute_type="int8")
    except Exception as exc:
        LOG.warning(f"Could not load whisper tiny for reference transcription: {exc}")
        whisper_tiny = None

    out: Dict[str, Dict[str, Any]] = {}
    try:
        for spk in speakers:
            spk_dir = profiles_dir / spk
            spk_dir.mkdir(parents=True, exist_ok=True)
            
            all_chunks = _collect_speaker_chunks(mono, sr, diarized_segments, spk)
            if not all_chunks:
                LOG.warning(f"  {spk}: no valid speech chunks found")
                continue

            # We want to pick the SINGLE BEST profile for stability.
            # We'll call it 'primary'.
            spk_profiles = {}
            
            # Display candidate scores for diagnostics
            LOG.info(f"  {spk} reference candidates:")
            for i, c in enumerate(all_chunks[:5]):
                LOG.info(f"    #{i+1}: score={c.score:.1f} dur={c.duration_s:.1f}s snr={c.snr:.1f}")

            # Best chunk
            ch = all_chunks[0]
            profile_name = "primary"
            prof_dir = spk_dir / profile_name
            prof_dir.mkdir(parents=True, exist_ok=True)

            # Build the reference. If the best chunk is short, try to
            # append other high-scoring chunks to reach target_seconds.
            ref_chunks = [ch]
            curr_dur = ch.duration_s
            
            if curr_dur < target_seconds:
                for other in all_chunks:
                    if other in ref_chunks: continue
                    if curr_dur + other.duration_s > max_seconds: continue
                    ref_chunks.append(other)
                    curr_dur += other.duration_s
                    if curr_dur >= target_seconds:
                        break

            # Concatenate with small gaps
            gap = int(0.06 * sr)
            ref_parts = []
            for j, rch in enumerate(ref_chunks):
                clip = mono[rch.start_sample:rch.end_sample]
                if j > 0:
                    ref_parts.append(np.zeros(gap, dtype="float32"))
                ref_parts.append(clip)
            
            ref_audio = np.concatenate(ref_parts)
            ref_audio = rms_normalize(ref_audio, target_dbfs=-20.0)
            ref_path = prof_dir / "reference.wav"
            write_wav(ref_path, ref_audio, sr)

            transcript = _transcribe_reference(ref_path, source_language, model=whisper_tiny)
            transcript_path = prof_dir / "transcript.txt"
            transcript_path.write_text(transcript + "\n", encoding="utf-8")
            word_count = len(transcript.split())

            duration_s = ref_audio.shape[0] / sr
            metadata = {
                "speaker_id": spk,
                "profile_id": profile_name,
                "score": round(ch.score, 2),
                "reference_duration": round(duration_s, 3),
                "segments_used": len(ref_chunks),
                "transcript_words": word_count,
                "reference_path": str(ref_path),
                "transcript_path": str(transcript_path),
            }
            meta_path = prof_dir / "metadata.json"
            meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

            spk_profiles[profile_name] = {
                "reference": str(ref_path),
                "transcript": str(transcript_path),
                "metadata": str(meta_path),
                "duration_s": duration_s,
                "word_count": word_count,
                "transcript_text": transcript,
                "score": ch.score,
            }

            # Flat alias for backwards compatibility
            if flat_dir is not None:
                flat = flat_dir / f"{spk}.wav"
                shutil.copy2(ref_path, flat)

            if spk_profiles:
                out[spk] = {"profiles": spk_profiles}
                LOG.info(f"  {spk}: primary profile selected (score={ch.score:.1f})")
    finally:
        if whisper_tiny:
            try:
                del whisper_tiny
            except Exception:
                pass
            free_vram()

    return out


class SampleStage:
    name = "samples"
    inputs: List[str] = ["media/speech.wav", "diarization/segments.json"]
    outputs: List[str] = [
        "speakers/speaker_01/primary.wav",
        "speakers/speaker_01/primary.txt",
        "speakers/speaker_01/embedding.npy",
        "speakers/speaker_01/metadata.json",
        "speakers/speaker_01/candidates/candidate_01.wav",
        "speakers/speaker_01/candidates/candidate_01.txt",
        "speakers/speaker_01/candidates/candidate_01.score.json",
    ]
    editable_outputs: List[str] = [
        "speakers/speaker_01/primary.wav",
        "speakers/speaker_01/primary.txt",
    ]
    derived_outputs: List[str] = []
    config_fields: List[str] = ["target_seconds", "max_seconds"]

    def __init__(
        self,
        workdir: Path,
        target_seconds: float = 10.0,
        max_seconds: float = 15.0,
        subdir: str | None = None,
    ):
        self.workdir = Path(workdir)
        if subdir:
            self.workdir = self.workdir / subdir
        self.target_seconds = target_seconds
        self.max_seconds = max_seconds
        self.subdir = subdir

    def output_paths(self) -> List[Path]:
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

        # Backwards-compatible mapping
        speaker_samples = {spk: p["profiles"]["primary"]["reference"] for spk, p in profiles.items() if "primary" in p["profiles"]}
        ref_transcripts = {spk: p["profiles"]["primary"]["transcript_text"] for spk, p in profiles.items() if "primary" in p["profiles"]}

        LOG.info("=" * 60)
        LOG.info(f"Voice profiles built: {len(profiles)}")
        for spk in sorted(profiles):
            p = profiles[spk]["profiles"]["primary"]
            LOG.info(f"  {spk}: dur={p['duration_s']:.2f}s score={p['score']:.1f} [SELECTED]")
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
