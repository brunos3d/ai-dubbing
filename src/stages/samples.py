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

def _frame_rms(audio: np.ndarray, sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0):
    """Return (rms_per_frame, hop_samples, frame_samples)."""
    win = max(1, int(frame_ms / 1000.0 * sr))
    hop = max(1, int(hop_ms / 1000.0 * sr))
    if audio.shape[0] < win:
        return np.array([np.sqrt(np.mean(audio ** 2))], dtype="float32"), hop, win
    n = 1 + (audio.shape[0] - win) // hop
    rms = np.empty(n, dtype="float32")
    for i in range(n):
        seg = audio[i * hop:i * hop + win]
        rms[i] = np.sqrt(np.mean(seg ** 2))
    return rms, hop, win


def _voiced_frame_mask(audio: np.ndarray, sr: int, ratio: float = 0.15) -> np.ndarray:
    """Boolean per-frame voiced mask via a peak-relative energy threshold.

    The threshold is a fraction of a robust peak (95th-percentile frame RMS),
    NOT of the signal's own low percentile — an adaptive noise-floor threshold
    collapses to "nothing is voiced" once a reference is already dense, which
    is exactly the dense, silence-trimmed signal we want to keep.
    """
    rms, _, _ = _frame_rms(audio, sr)
    if rms.size == 0:
        return np.zeros(0, dtype=bool)
    peak = float(np.percentile(rms, 95))
    if peak < 1e-6:
        return np.zeros_like(rms, dtype=bool)
    return rms > ratio * peak


def _voiced_frac(audio: np.ndarray, sr: int) -> float:
    """Fraction of frames that are voiced (speech-active)."""
    mask = _voiced_frame_mask(audio, sr)
    return float(mask.mean()) if mask.size else 0.0


def trim_to_voiced(
    audio: np.ndarray,
    sr: int,
    keep_pause_s: float = 0.12,
    edge_pad_s: float = 0.03,
) -> np.ndarray:
    """Drop silence/ambience-only stretches, keeping dense voiced speech.

    Inter-word pauses up to ``keep_pause_s`` are preserved so the reference
    still sounds like natural speech; longer silences (and ambience-only
    regions, which read as unvoiced) are removed. A short ``edge_pad_s`` of
    context is kept around each voiced run to avoid clipping onsets.
    """
    mask = _voiced_frame_mask(audio, sr)
    if not mask.any():
        return audio
    _, hop, win = _frame_rms(audio, sr)
    # Expand frame mask to a per-sample keep mask.
    keep = np.zeros(audio.shape[0], dtype=bool)
    pad = int(edge_pad_s * sr)
    for i, v in enumerate(mask):
        if not v:
            continue
        s = max(0, i * hop - pad)
        e = min(audio.shape[0], i * hop + win + pad)
        keep[s:e] = True
    # Bridge short unkept gaps (<= keep_pause_s) so speech isn't choppy.
    bridge = int(keep_pause_s * sr)
    if bridge > 0:
        idx = np.flatnonzero(keep)
        if idx.size:
            prev = idx[0]
            for j in idx[1:]:
                if 0 < j - prev <= bridge:
                    keep[prev:j] = True
                prev = j
    out = audio[keep]
    return out if out.size else audio


def denoise_spectral_gate(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    reduction_db: float = 12.0,
) -> np.ndarray:
    """Conservative spectral-gating denoise.

    Estimates a noise magnitude profile from the quietest 10% of frames and
    attenuates (never fully removes) spectral content near that floor. The
    attenuation is capped at ``reduction_db`` so the speaker's timbre is
    preserved — over-aggressive subtraction would corrupt the very identity
    we are trying to clone.
    """
    if audio.shape[0] < n_fft:
        return audio
    try:
        import librosa
    except Exception:
        return audio
    hop = n_fft // 4
    S = librosa.stft(audio.astype("float32"), n_fft=n_fft, hop_length=hop)
    mag, phase = np.abs(S), np.angle(S)
    frame_energy = mag.sum(axis=0)
    if frame_energy.size < 4:
        return audio
    quiet = frame_energy <= np.percentile(frame_energy, 10)
    if not quiet.any():
        return audio
    # Pass-through guard: only denoise when a genuine noise floor sits well
    # below the speech. If the quietest frames are nearly as loud as the
    # median (a dense, already-clean reference), spectral subtraction would
    # estimate "noise" from the voice itself and gut the timbre.
    loud_e = float(np.percentile(frame_energy, 75))
    quiet_e = float(np.median(frame_energy[quiet]))
    if quiet_e <= 1e-9 or loud_e / quiet_e < 3.0:
        return audio
    noise_mag = np.median(mag[:, quiet], axis=1, keepdims=True)
    floor = 10.0 ** (-reduction_db / 20.0)
    # Soft Wiener-style gain: full pass well above noise, floored attenuation near it.
    snr = mag / (noise_mag + 1e-9)
    gain = snr ** 2 / (snr ** 2 + 1.0)
    gain = np.clip(gain, floor, 1.0)
    out = librosa.istft(gain * mag * np.exp(1j * phase), hop_length=hop, length=audio.shape[0])
    return out.astype("float32")


def purify_reference(audio: np.ndarray, sr: int) -> np.ndarray:
    """Maximum-purity pass for a cloning reference: trim silence, then denoise.

    Order matters: trimming first removes the silence/ambience pockets that
    would otherwise dominate the noise-profile estimate, so the subsequent
    gentle spectral gate targets residual room/crowd bleed under the voice.
    """
    trimmed = trim_to_voiced(audio, sr)
    if trimmed.shape[0] < int(0.3 * sr):
        trimmed = audio
    return denoise_spectral_gate(trimmed, sr)


@dataclass
class SpeechChunk:
    """A voiced region inside a diarized segment with quality metrics."""
    speaker: str
    start_sample: int        # absolute position in the speech.wav array
    end_sample: int
    duration_s: float
    snr: float = 0.0         # estimated signal-to-noise ratio
    rms: float = 0.0         # energy
    voiced_frac: float = 1.0  # fraction of the chunk that is voiced speech
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

    # 5. Voiced density (purity): a window that is mostly silence or ambience
    # makes a diluted cloning reference. Reward dense voiced speech and
    # penalise sparse chunks hard, so the reference is the target voice — not
    # the pauses around it (Objective #1).
    vf = chunk.voiced_frac
    if vf >= 0.7:
        voiced_score = 25.0
    elif vf >= 0.4:
        voiced_score = 25.0 * (vf - 0.4) / 0.3
    else:
        voiced_score = -20.0  # actively avoid sparse/contaminated windows

    return snr_score + dur_score + rms_score + continuity_bonus + voiced_score


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
            rms=rms,
            voiced_frac=_voiced_frac(clip, sr),
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
            from ..utils.asr import resolve_ref_asr
            size, device, compute = resolve_ref_asr()
            LOG.info(f"Transcribing reference {audio_path.name} with faster-whisper ({size} on {device})")
            model = WhisperModel(size, device=device, compute_type=compute)
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

    # Pre-load the reference-ASR model once for all speaker references.
    # Runs on GPU with a stronger model when available (Objective #5) so the
    # clone prompt (ref_text) is faithful rather than tiny's best guess.
    from faster_whisper import WhisperModel
    from ..utils.asr import resolve_ref_asr
    _ref_size, _ref_dev, _ref_compute = resolve_ref_asr()
    try:
        whisper_tiny = WhisperModel(_ref_size, device=_ref_dev, compute_type=_ref_compute)
        LOG.info(f"Reference ASR: {_ref_size} on {_ref_dev} ({_ref_compute})")
    except Exception as exc:
        LOG.warning(f"Could not load reference ASR ({_ref_size}/{_ref_dev}): {exc}; falling back to tiny/cpu")
        try:
            whisper_tiny = WhisperModel("tiny", device="cpu", compute_type="int8")
        except Exception:
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
            # Maximum-purity pass: strip silence/ambience pockets and gently
            # denoise residual room/crowd bleed so the embedding sees the
            # target voice only (Objectives #1/#2).
            voiced_before = _voiced_frac(ref_audio, sr)
            ref_audio = purify_reference(ref_audio, sr)
            voiced_after = _voiced_frac(ref_audio, sr)
            ref_audio = rms_normalize(ref_audio, target_dbfs=-20.0)
            ref_path = prof_dir / "reference.wav"
            write_wav(ref_path, ref_audio, sr)
            LOG.info(
                f"  {spk}: purified reference voiced {voiced_before:.2f} -> "
                f"{voiced_after:.2f} ({ref_audio.shape[0] / sr:.1f}s)"
            )

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
                "voiced_frac": round(voiced_after, 3),
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
