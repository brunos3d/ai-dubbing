"""Measurable quality metrics for a reconstructed dub vs. its source.

Every metric returns a float in ``[0, 1]`` where **higher is better**, or
``None`` when it cannot be computed from the artifacts on disk. The optimizer
maximises a weighted :func:`score_from_metrics` composite of whatever metrics
are available, so a missing metric degrades gracefully (it is dropped and the
remaining weights renormalise) rather than crashing the run.

All metrics read only artifacts the pipeline already produces in a workspace:

* ``output/reconstructed_speech.wav`` — the dub's speech track (segments placed
  at their original start times on a silent timeline).
* ``media/speech.wav`` — the original separated speech (the same-language
  ground truth).
* ``aligned_manifest.json`` — per-segment timing, prosody and paths.
* ``speaker_profiles/<spk>/primary/reference.wav`` — the per-speaker reference.

The same-language benchmark (EN→EN) is what makes acoustic comparison valid:
the words are identical, so spectral envelope, energy contour and speaker
timbre can be compared directly against the source.

Implementation is numpy + (optional) librosa — CPU-only, no torch, no network.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Default composite weights. They sum to 1.0 and encode the task's quality
# priorities (timing, continuity, prosody, endings, speaker identity). They are
# a plain dict so a caller — or the optimizer config — can override any subset.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "timing_accuracy": 0.15,
    "slot_fit": 0.10,
    "speaker_similarity": 0.15,
    "speaker_consistency": 0.08,
    "ending_quality": 0.08,
    "continuity": 0.10,
    "prosodic_similarity": 0.09,
    "reconstruction_quality": 0.08,
    "ambience_preservation": 0.09,
    "reaction_preservation": 0.08,
}


@dataclass
class MetricResult:
    """The per-metric values, the composite score, and diagnostic detail."""

    metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    composite: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "composite": (round(self.composite, 5) if self.composite is not None else None),
            "metrics": {
                k: (round(v, 5) if v is not None else None)
                for k, v in self.metrics.items()
            },
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _pearson(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.size < 3 or b.size < 3 or a.size != b.size:
        return None
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _read_mono(path: Path) -> Tuple[Optional[np.ndarray], int]:
    """Read a wav as a 1-D mono float array; return ``(None, 0)`` on failure."""
    try:
        from ..utils.audio import read_wav

        data, sr = read_wav(path, dtype="float32", always_2d=True)
        if data.ndim == 2:
            data = data.mean(axis=0)
        return np.asarray(data, dtype=np.float32).reshape(-1), int(sr)
    except Exception:  # noqa: BLE001 - any read error → metric unavailable
        return None, 0


def _mfcc_feature(audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
    """A compact, comparable timbre descriptor (mean+std MFCC at 16 kHz)."""
    if audio is None or audio.size < int(0.1 * sr):
        return None
    try:
        import librosa  # type: ignore

        target_sr = 16000
        if sr != target_sr:
            audio = librosa.resample(audio.astype("float32"), orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        mfcc = librosa.feature.mfcc(y=audio.astype("float32"), sr=sr, n_mfcc=20)
        return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]).astype("float32")
    except Exception:  # noqa: BLE001
        return None


def _logmel_envelope(audio: np.ndarray, sr: int, n_mels: int = 40) -> Optional[np.ndarray]:
    """Time-averaged log-mel spectrum — a coarse spectral-envelope fingerprint."""
    if audio is None or audio.size < int(0.2 * sr):
        return None
    try:
        import librosa  # type: ignore

        target_sr = 16000
        if sr != target_sr:
            audio = librosa.resample(audio.astype("float32"), orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        mel = librosa.feature.melspectrogram(y=audio.astype("float32"), sr=sr, n_mels=n_mels)
        return np.log(mel.mean(axis=1) + 1e-9).astype("float32")
    except Exception:  # noqa: BLE001
        return None


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0


def _db(x: float) -> float:
    return 20.0 * math.log10(x) if x > 1e-9 else -120.0


# ---------------------------------------------------------------------------
# Manifest-based metrics (no audio needed)
# ---------------------------------------------------------------------------


def _speech_entries(manifest: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Synthesized speech segments only (skip preserved non-speech clips).

    Non-speech segments are byte-for-byte copies of the source, so they would
    trivially score perfectly on timing/identity and dilute the signal.
    """
    return [e for e in manifest if not e.get("is_non_speech")]


def timing_accuracy(manifest: List[Dict[str, Any]]) -> Tuple[Optional[float], Dict[str, Any]]:
    """How close each segment's final duration is to its source slot.

    Score per segment = ``max(0, 1 - rel_err / 0.5)`` (0 at ≥50 % error),
    averaged. Uses ``aligned_duration`` (post-stretch) against the source slot.
    """
    rows = _speech_entries(manifest)
    errs: List[float] = []
    for e in rows:
        target = float(e.get("target_duration") or e.get("original_duration") or 0.0)
        cur = float(e.get("aligned_duration") or e.get("generated_duration") or 0.0)
        if target <= 0:
            continue
        rel = abs(cur - target) / target
        errs.append(_clamp(1.0 - rel / 0.5))
    if not errs:
        return None, {}
    return float(np.mean(errs)), {
        "n_segments": len(errs),
        "mean_abs_rel_error": round(float(np.mean([
            abs(float(e.get("aligned_duration") or 0.0) - float(e.get("target_duration") or e.get("original_duration") or 0.0))
            / max(0.05, float(e.get("target_duration") or e.get("original_duration") or 0.0))
            for e in rows if float(e.get("target_duration") or e.get("original_duration") or 0.0) > 0
        ])), 4),
    }


def slot_fit(manifest: List[Dict[str, Any]]) -> Tuple[Optional[float], Dict[str, Any]]:
    """Penalise overruns — speech that runs past its slot bleeds into the next
    turn (the most audible timing failure). Underruns leave a natural pause and
    are not penalised here.

    Score per segment = ``1 - clamp(overrun / target / 0.3)``, averaged.
    """
    rows = _speech_entries(manifest)
    scores: List[float] = []
    n_overrun = 0
    for e in rows:
        target = float(e.get("target_duration") or e.get("original_duration") or 0.0)
        cur = float(e.get("aligned_duration") or e.get("generated_duration") or 0.0)
        if target <= 0:
            continue
        overrun = max(0.0, cur - target)
        if overrun > 0.05 * target:
            n_overrun += 1
        scores.append(1.0 - _clamp(overrun / target / 0.3))
    if not scores:
        return None, {}
    return float(np.mean(scores)), {
        "n_overrunning": n_overrun,
        "overrun_fraction": round(n_overrun / len(scores), 4),
    }


def continuity(manifest: List[Dict[str, Any]]) -> Tuple[Optional[float], Dict[str, Any]]:
    """Penalise temporal collisions between consecutive segments.

    When a segment's placed audio (start + final duration) overlaps the next
    segment's start, two turns play on top of each other — an abrupt, muddled
    transition. Score = ``1 - clamp(mean overlap fraction / 0.2)``.
    """
    entries = sorted(manifest, key=lambda e: float(e.get("start", 0.0)))
    overlaps: List[float] = []
    n_collisions = 0
    for cur, nxt in zip(entries, entries[1:]):
        cur_start = float(cur.get("start", 0.0))
        dur = float(cur.get("aligned_duration") or cur.get("generated_duration") or 0.0)
        cur_end = cur_start + dur
        nxt_start = float(nxt.get("start", 0.0))
        gap = nxt_start - cur_end
        if gap < 0:  # collision
            n_collisions += 1
            overlaps.append(min(1.0, -gap / max(0.2, dur)))
        else:
            overlaps.append(0.0)
    if not overlaps:
        return None, {}
    mean_overlap = float(np.mean(overlaps))
    return 1.0 - _clamp(mean_overlap / 0.2), {
        "n_collisions": n_collisions,
        "mean_overlap_fraction": round(mean_overlap, 4),
    }


# ---------------------------------------------------------------------------
# Audio-based metrics
# ---------------------------------------------------------------------------


def _segment_audio(entry: Dict[str, Any]) -> Tuple[Optional[np.ndarray], int]:
    path = entry.get("aligned_path") or entry.get("path")
    if not path:
        return None, 0
    p = Path(path)
    if not p.exists():
        return None, 0
    return _read_mono(p)


def speaker_metrics(
    manifest: List[Dict[str, Any]], workspace_root: Path
) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
    """Return ``(speaker_similarity, speaker_consistency, detail)``.

    * **similarity** — cosine between each generated segment's MFCC timbre and
      its speaker's *reference* timbre, mapped to ``[0, 1]`` and averaged. High
      ⇒ the cloned voice matches the real speaker (identity preservation).
    * **consistency** — mean intra-speaker cohesion (pairwise cosine of a
      speaker's own segments). High ⇒ a speaker sounds like one stable person
      across the dub (no drift / no attribution swaps).
    """
    rows = _speech_entries(manifest)
    if not rows:
        return None, None, {}

    # Per-speaker reference features (cached).
    ref_feat: Dict[str, Optional[np.ndarray]] = {}

    def _ref_for(spk: str) -> Optional[np.ndarray]:
        if spk in ref_feat:
            return ref_feat[spk]
        ref_path = workspace_root / "speaker_profiles" / spk / "primary" / "reference.wav"
        audio, sr = _read_mono(ref_path) if ref_path.exists() else (None, 0)
        ref_feat[spk] = _mfcc_feature(audio, sr) if audio is not None else None
        return ref_feat[spk]

    sims: List[float] = []
    by_speaker: Dict[str, List[np.ndarray]] = {}
    for e in rows:
        spk = e.get("speaker", "")
        audio, sr = _segment_audio(e)
        feat = _mfcc_feature(audio, sr) if audio is not None else None
        if feat is None:
            continue
        by_speaker.setdefault(spk, []).append(feat)
        ref = _ref_for(spk)
        if ref is not None:
            sims.append((_cosine(feat, ref) + 1.0) / 2.0)

    similarity = float(np.mean(sims)) if sims else None

    # Intra-speaker cohesion.
    cohesions: List[float] = []
    for spk, feats in by_speaker.items():
        if len(feats) < 2:
            continue
        pair_sims = []
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                pair_sims.append((_cosine(feats[i], feats[j]) + 1.0) / 2.0)
        if pair_sims:
            cohesions.append(float(np.mean(pair_sims)))
    consistency = float(np.mean(cohesions)) if cohesions else None

    return similarity, consistency, {
        "n_compared": len(sims),
        "n_speakers": len(by_speaker),
    }


def ending_quality(manifest: List[Dict[str, Any]]) -> Tuple[Optional[float], Dict[str, Any]]:
    """Detect clipped / abruptly-cut segment endings.

    A naturally-finished utterance decays into quiet at its tail; a clipped one
    is still loud at the very last samples. We compare the RMS of the final
    ~40 ms to the clip's overall RMS — a tail much louder than the body signals
    a hard cut. Score per segment = ``1 - clamp((tail_ratio - 0.7) / 1.0)``.
    """
    rows = _speech_entries(manifest)
    scores: List[float] = []
    for e in rows:
        audio, sr = _segment_audio(e)
        if audio is None or sr <= 0 or audio.size < int(0.1 * sr):
            continue
        tail_n = max(1, int(0.04 * sr))
        tail_rms = _rms(audio[-tail_n:])
        body_rms = _rms(audio)
        if body_rms < 1e-6:
            continue
        ratio = tail_rms / body_rms
        scores.append(1.0 - _clamp((ratio - 0.7) / 1.0))
    if not scores:
        return None, {}
    return float(np.mean(scores)), {"n_segments": len(scores)}


def prosodic_similarity(manifest: List[Dict[str, Any]]) -> Tuple[Optional[float], Dict[str, Any]]:
    """Correlate the source energy contour with the generated one.

    The generate stage records each segment's *source* prosody under
    ``prosody.energy_db``. We measure the *generated* segment's energy the same
    way and correlate the two sequences across segments. High correlation ⇒
    emphasis/de-emphasis across the conversation is preserved (a loud, excited
    line stays loud in the dub).
    """
    rows = _speech_entries(manifest)
    src: List[float] = []
    gen: List[float] = []
    for e in rows:
        pros = e.get("prosody") or {}
        src_db = pros.get("energy_db")
        if src_db is None:
            continue
        audio, sr = _segment_audio(e)
        if audio is None or audio.size == 0:
            continue
        src.append(float(src_db))
        gen.append(_db(_rms(audio)))
    r = _pearson(np.asarray(src), np.asarray(gen))
    if r is None:
        return None, {"n_segments": len(src)}
    return (r + 1.0) / 2.0, {"n_segments": len(src), "pearson_r": round(r, 4)}


def reconstruction_quality(workspace_root: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    """Global comparison of the reconstructed dub speech vs the source speech.

    Two cheap, robust components, averaged:

    * **duration match** — total length ratio ``min/max`` (the dub should span
      roughly the same time as the source).
    * **spectral-envelope correlation** — Pearson correlation of the
      time-averaged log-mel spectra. For same-language dubbing the words match,
      so the overall spectral shape should track the source closely.
    """
    dub_path = workspace_root / "output" / "reconstructed_speech.wav"
    src_path = workspace_root / "media" / "speech.wav"
    if not dub_path.exists() or not src_path.exists():
        return None, {}
    dub, dsr = _read_mono(dub_path)
    src, ssr = _read_mono(src_path)
    if dub is None or src is None:
        return None, {}

    dub_s = dub.size / dsr if dsr else 0.0
    src_s = src.size / ssr if ssr else 0.0
    dur_match = (min(dub_s, src_s) / max(dub_s, src_s)) if max(dub_s, src_s) > 0 else 0.0

    env_dub = _logmel_envelope(dub, dsr)
    env_src = _logmel_envelope(src, ssr)
    env_score: Optional[float] = None
    if env_dub is not None and env_src is not None and env_dub.shape == env_src.shape:
        r = _pearson(env_dub, env_src)
        if r is not None:
            env_score = (r + 1.0) / 2.0

    components = [dur_match] + ([env_score] if env_score is not None else [])
    score = float(np.mean(components)) if components else None
    return score, {
        "dub_seconds": round(dub_s, 2),
        "source_seconds": round(src_s, 2),
        "duration_match": round(dur_match, 4),
        "envelope_correlation": (round(env_score, 4) if env_score is not None else None),
    }


def ambience_preservation(workspace_root: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    """Measure how well the dub preserves the original ambience/reactions.

    Compares the original audio (with speech masked out) to the final mixed
    audio (with synthesized speech masked out). High correlation in the
    non-speech regions indicates good ambience preservation.

    Components:
    * **energy envelope correlation** — how well the energy contour matches
      in non-speech regions
    * **spectral similarity** — how well the spectral content matches
    """
    original_path = workspace_root / "media" / "original_audio.wav"
    final_path = workspace_root / "output" / "final_audio.wav"
    manifest_path = workspace_root / "aligned_manifest.json"

    if not original_path.exists() or not final_path.exists():
        return None, {}

    orig, osr = _read_mono(original_path)
    final, fsr = _read_mono(final_path)

    if orig is None or final is None:
        return None, {}

    # Load manifest to get speech segment timings
    speech_segments = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            speech_segments = [
                (float(e.get("start", 0)), float(e.get("end", 0)))
                for e in manifest
                if not e.get("is_non_speech")
            ]
        except Exception:
            pass

    # Resample to common rate for comparison
    target_sr = min(osr, fsr, 16000)
    try:
        import librosa
        if osr != target_sr:
            orig = librosa.resample(orig.astype("float32"), orig_sr=osr, target_sr=target_sr)
        if fsr != target_sr:
            final = librosa.resample(final.astype("float32"), orig_sr=fsr, target_sr=target_sr)
        sr = target_sr
    except Exception:
        sr = osr

    # Build non-speech mask
    n = min(orig.size, final.size)
    orig = orig[:n]
    final = final[:n]

    is_speech = np.zeros(n, dtype=bool)
    for start, end in speech_segments:
        s = max(0, int(start * sr))
        e = min(n, int(end * sr))
        is_speech[s:e] = True

    non_speech_mask = ~is_speech
    if non_speech_mask.sum() < int(0.5 * sr):  # Need at least 0.5s of non-speech
        return None, {"reason": "insufficient_non_speech"}

    orig_nonspeech = orig[non_speech_mask]
    final_nonspeech = final[non_speech_mask]

    # Energy envelope correlation
    frame_ms = 50.0
    win = max(1, int(frame_ms / 1000.0 * sr))
    hop = win // 2

    def _frame_energy(x):
        if x.size < win:
            return np.array([np.sqrt(np.mean(x ** 2))])
        n_frames = 1 + (x.size - win) // hop
        energy = np.empty(n_frames, dtype="float32")
        for i in range(n_frames):
            energy[i] = np.sqrt(np.mean(x[i * hop:i * hop + win] ** 2))
        return energy

    orig_energy = _frame_energy(orig_nonspeech)
    final_energy = _frame_energy(final_nonspeech)

    min_len = min(orig_energy.size, final_energy.size)
    if min_len < 3:
        return None, {"reason": "too_short"}

    energy_corr = _pearson(orig_energy[:min_len], final_energy[:min_len])
    energy_score = (energy_corr + 1.0) / 2.0 if energy_corr is not None else 0.5

    # Spectral similarity (log-mel envelope)
    orig_env = _logmel_envelope(orig_nonspeech, sr)
    final_env = _logmel_envelope(final_nonspeech, sr)

    spectral_score: Optional[float] = None
    if orig_env is not None and final_env is not None and orig_env.shape == final_env.shape:
        r = _pearson(orig_env, final_env)
        if r is not None:
            spectral_score = (r + 1.0) / 2.0

    components = [energy_score]
    if spectral_score is not None:
        components.append(spectral_score)

    score = float(np.mean(components))
    return score, {
        "energy_correlation": round(energy_score, 4),
        "spectral_similarity": (round(spectral_score, 4) if spectral_score is not None else None),
        "non_speech_seconds": round(non_speech_mask.sum() / sr, 2),
    }


def reaction_preservation(workspace_root: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    """Measure how well audience reactions (laughter, applause) are preserved.

    Detects reaction events in the original audio and checks if they appear
    in the final mixed audio at similar times and intensities.

    Components:
    * **occurrence match** — are reactions present at similar times?
    * **intensity match** — are reaction intensities similar?
    """
    original_path = workspace_root / "media" / "original_audio.wav"
    final_path = workspace_root / "output" / "final_audio.wav"
    manifest_path = workspace_root / "aligned_manifest.json"

    if not original_path.exists() or not final_path.exists():
        return None, {}

    orig, osr = _read_mono(original_path)
    final, fsr = _read_mono(final_path)

    if orig is None or final is None:
        return None, {}

    # Load speech segments
    speech_segments = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            speech_segments = [
                {"start": float(e.get("start", 0)), "end": float(e.get("end", 0))}
                for e in manifest
                if not e.get("is_non_speech")
            ]
        except Exception:
            pass

    if not speech_segments:
        return None, {"reason": "no_speech_segments"}

    # Detect reactions in original
    try:
        from ..utils.audio_events import detect_reactions
        orig_reactions = detect_reactions(orig, osr, speech_segments)
        final_reactions = detect_reactions(final, fsr, speech_segments)
    except Exception:
        return None, {"reason": "detection_failed"}

    if not orig_reactions:
        # No reactions in original - perfect preservation trivially
        return 1.0, {"n_original": 0, "n_final": 0, "note": "no_reactions_in_source"}

    # Match reactions by temporal overlap
    matched = 0
    intensity_scores = []

    for orig_r in orig_reactions:
        # Find best matching final reaction
        best_overlap = 0.0
        best_final = None
        for final_r in final_reactions:
            overlap_start = max(orig_r.start, final_r.start)
            overlap_end = min(orig_r.end, final_r.end)
            if overlap_end > overlap_start:
                overlap = overlap_end - overlap_start
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_final = final_r

        if best_overlap > 0.1:  # At least 100ms overlap
            matched += 1
            if best_final is not None:
                # Compare intensities
                intensity_diff = abs(orig_r.intensity - best_final.intensity)
                intensity_scores.append(1.0 - min(1.0, intensity_diff))

    occurrence_score = matched / len(orig_reactions) if orig_reactions else 0.0
    intensity_score = float(np.mean(intensity_scores)) if intensity_scores else 0.5

    # Combined score: 60% occurrence, 40% intensity
    score = 0.6 * occurrence_score + 0.4 * intensity_score

    return score, {
        "n_original": len(orig_reactions),
        "n_final": len(final_reactions),
        "n_matched": matched,
        "occurrence_score": round(occurrence_score, 4),
        "intensity_score": round(intensity_score, 4),
    }


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


def score_from_metrics(
    metrics: Dict[str, Optional[float]],
    weights: Optional[Dict[str, float]] = None,
) -> Optional[float]:
    """Weighted average of available metrics, renormalised over the present
    (non-``None``) ones. Returns ``None`` if nothing is available."""
    weights = weights or DEFAULT_WEIGHTS
    num = 0.0
    den = 0.0
    for name, value in metrics.items():
        if value is None:
            continue
        w = weights.get(name, 0.0)
        if w <= 0:
            continue
        num += w * float(value)
        den += w
    if den <= 0:
        return None
    return num / den


def compute_metrics(
    workspace_root: Path,
    *,
    weights: Optional[Dict[str, float]] = None,
    manifest: Optional[List[Dict[str, Any]]] = None,
) -> MetricResult:
    """Compute every metric for a finished (reconstructed) workspace.

    ``manifest`` may be supplied directly (used in tests); otherwise it is read
    from ``aligned_manifest.json``. Missing artifacts yield ``None`` metrics,
    which the composite drops and renormalises around.
    """
    workspace_root = Path(workspace_root)
    if manifest is None:
        man_path = workspace_root / "aligned_manifest.json"
        if man_path.exists():
            try:
                manifest = json.loads(man_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = []
        else:
            manifest = []

    result = MetricResult()

    t_acc, t_det = timing_accuracy(manifest)
    s_fit, s_det = slot_fit(manifest)
    cont, c_det = continuity(manifest)
    sim, cons, sp_det = speaker_metrics(manifest, workspace_root)
    end, e_det = ending_quality(manifest)
    pros, p_det = prosodic_similarity(manifest)
    recon, r_det = reconstruction_quality(workspace_root)
    ambience, amb_det = ambience_preservation(workspace_root)
    reaction, react_det = reaction_preservation(workspace_root)

    result.metrics = {
        "timing_accuracy": t_acc,
        "slot_fit": s_fit,
        "continuity": cont,
        "speaker_similarity": sim,
        "speaker_consistency": cons,
        "ending_quality": end,
        "prosodic_similarity": pros,
        "reconstruction_quality": recon,
        "ambience_preservation": ambience,
        "reaction_preservation": reaction,
    }
    result.details = {
        "timing_accuracy": t_det,
        "slot_fit": s_det,
        "continuity": c_det,
        "speaker": sp_det,
        "ending_quality": e_det,
        "prosodic_similarity": p_det,
        "reconstruction_quality": r_det,
        "ambience_preservation": amb_det,
        "reaction_preservation": react_det,
        "n_manifest_entries": len(manifest),
    }
    result.composite = score_from_metrics(result.metrics, weights)
    return result
