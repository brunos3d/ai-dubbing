"""Stage 3 - Speaker diarization.

Primary: pyannote.audio (requires ``HF_TOKEN`` to access the gated models).
Fallback: VAD + MFCC clustering using Silero VAD, which is fully open.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.audio import read_wav
from ..utils.logging import get_logger, stage_banner
from ..utils.vram import free_vram, log_vram

LOG = get_logger("ai-dubbing.diarize")


def _have_pyannote_auth(hf_token: Optional[str]) -> bool:
    return bool(hf_token or os.environ.get("HF_TOKEN"))


def _make_pipeline(model_id: str, hf_token: Optional[str]):
    from pyannote.audio import Pipeline

    use_auth = hf_token or os.environ.get("HF_TOKEN")
    try:
        return Pipeline.from_pretrained(model_id, use_auth_token=use_auth)
    except TypeError:
        return Pipeline.from_pretrained(model_id, token=use_auth)


def _annotation_to_segments(annotation, min_duration: float = 0.3) -> List[Dict[str, Any]]:
    """Convert pyannote Annotation to a list of {speaker, start, end} dicts."""
    out: List[Dict[str, Any]] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        start = float(turn.start)
        end = float(turn.end)
        if end - start < min_duration:
            continue
        out.append({
            "speaker": str(speaker),
            "start": round(start, 3),
            "end": round(end, 3),
        })
    out.sort(key=lambda r: r["start"])
    return out


def _relabel_speakers(segments: List[Dict[str, Any]], prefix: str = "speaker_") -> List[Dict[str, Any]]:
    seen: Dict[str, str] = {}
    out: List[Dict[str, Any]] = []
    for seg in segments:
        spk = seg["speaker"]
        if spk not in seen:
            seen[spk] = f"{prefix}{len(seen) + 1:02d}"
        new = dict(seg)
        new["speaker"] = seen[spk]
        out.append(new)
    return out


def _merge_tiny_speakers(
    segments: List[Dict[str, Any]],
    min_total_duration: float = 2.0,
    min_segments: int = 2,
) -> List[Dict[str, Any]]:
    """Merge speakers with very little material into the dominant speaker."""
    if not segments:
        return []

    stats: Dict[str, Dict[str, Any]] = {}
    for seg in segments:
        spk = seg["speaker"]
        if spk not in stats:
            stats[spk] = {"duration": 0.0, "count": 0}
        stats[spk]["duration"] += seg["end"] - seg["start"]
        stats[spk]["count"] += 1

    if not stats:
        return segments

    dominant_speaker = max(stats.keys(), key=lambda k: stats[k]["duration"])
    to_merge = set()
    for spk, data in stats.items():
        if spk == dominant_speaker:
            continue
        if data["duration"] < min_total_duration or data["count"] < min_segments:
            LOG.info(f"Merging unreliable speaker {spk} ({data['duration']:.2f}s, {data['count']} segs) into {dominant_speaker}")
            to_merge.add(spk)

    if not to_merge:
        return segments

    out = []
    for seg in segments:
        new = dict(seg)
        if new["speaker"] in to_merge:
            new["speaker"] = dominant_speaker
        out.append(new)
    return out


# -----------------------------
# Fallback diarization
# -----------------------------

def _silero_vad_segments(
    audio: np.ndarray,
    sample_rate: int,
    min_seg_dur: float = 0.4,
    pad: float = 0.10,
) -> List[Tuple[float, float]]:
    import torch

    target_sr = 16000
    if audio.ndim == 2:
        mono = audio.mean(axis=0)
    else:
        mono = audio
    if sample_rate != target_sr:
        import librosa  # type: ignore

        mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=target_sr)
    wav = torch.from_numpy(mono.astype("float32"))
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
        verbose=False,
    )
    get_speech_timestamps = utils[0]
    try:
        ts = get_speech_timestamps(wav[0], model, sampling_rate=target_sr,
                                   min_speech_duration_ms=int(min_seg_dur * 1000),
                                   min_silence_duration_ms=300,
                                   return_seconds=True)
    except TypeError:
        ts = get_speech_timestamps(wav[0], model, sampling_rate=target_sr,
                                   return_seconds=True)
    out: List[Tuple[float, float]] = []
    for entry in ts:
        if isinstance(entry, dict):
            t0 = float(entry.get("start", 0.0)) - pad
            t1 = float(entry.get("end", 0.0)) + pad
        else:
            t0 = float(entry[0]) / target_sr - pad
            t1 = float(entry[1]) / target_sr + pad
        if t1 - t0 >= min_seg_dur:
            out.append((max(0.0, t0), t1))
    return out


def _speaker_embeddings(
    audio: np.ndarray,
    sample_rate: int,
    segments: List[Tuple[float, float]],
) -> np.ndarray:
    """Compute per-segment MFCC-based features (lightweight, dependency-free)."""
    if audio.ndim == 2:
        mono = audio.mean(axis=0)
    else:
        mono = audio
    try:
        import librosa  # type: ignore

        target_sr = 16000
        if sample_rate != target_sr:
            mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=target_sr)
            sr = target_sr
        else:
            sr = sample_rate
        feats = []
        for ts, te in segments:
            s0 = int(max(0, ts) * sr)
            s1 = int(min(mono.shape[0], te) * sr)
            if s1 - s0 < int(0.25 * sr):
                feats.append(np.zeros(20, dtype=np.float32))
                continue
            seg = mono[s0:s1]
            mfcc = librosa.feature.mfcc(y=seg.astype("float32"), sr=sr, n_mfcc=20)
            d1 = librosa.feature.delta(mfcc)
            feats.append(np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1), d1.mean(axis=1)]))
        return np.stack(feats)
    except ImportError:
        sr = sample_rate
        feats = []
        win = int(0.025 * sr)
        hop = int(0.010 * sr)
        for ts, te in segments:
            s0 = int(max(0, ts) * sr)
            s1 = int(min(mono.shape[0], te) * sr)
            if s1 - s0 < win:
                feats.append(np.zeros(40, dtype=np.float32))
                continue
            seg = mono[s0:s1]
            n_frames = (s1 - s0 - win) // hop + 1
            frames = np.lib.stride_tricks.as_strided(
                seg,
                shape=(n_frames, win),
                strides=(seg.strides[0] * hop, seg.strides[0]),
            )
            pow_spec = np.abs(np.fft.rfft(frames, n=512, axis=1)) ** 2
            mel = pow_spec.mean(axis=0)
            log_mel = np.log(mel + 1e-9)
            dct = np.zeros(40, dtype=np.float32)
            for k in range(40):
                dct[k] = np.sum(log_mel * np.cos(np.pi * k * (np.arange(log_mel.shape[0]) + 0.5) / log_mel.shape[0]))
            feats.append(dct)
        return np.stack(feats)


def _cluster_embeddings(
    feats: np.ndarray,
    num_speakers: int,
) -> np.ndarray:
    """Cluster per-segment embeddings.

    If ``num_speakers`` is <= 0 we auto-detect the number of clusters using a
    silhouette-driven sweep over a small range.  This avoids the previous
    failure where everything was forced into 2 clusters regardless of the
    actual diversity of voices in the recording.
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    if feats.shape[0] == 0:
        return np.zeros(0, dtype=int)
    norm = StandardScaler().fit_transform(feats)

    if num_speakers <= 0:
        best_k, best_score = 1, -2.0
        upper = min(8, feats.shape[0])
        for k in range(2, max(2, upper) + 1):
            try:
                lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(norm)
                if len(set(lab)) < 2:
                    continue
                score = silhouette_score(norm, lab)
            except Exception:
                continue
            if score > best_score:
                best_score = score
                best_k = k
        num_speakers = best_k
    if num_speakers < 1:
        num_speakers = 1
    if num_speakers > feats.shape[0]:
        num_speakers = feats.shape[0]
    if num_speakers == 1:
        return np.zeros(feats.shape[0], dtype=int)
    clusterer = AgglomerativeClustering(n_clusters=num_speakers, linkage="ward")
    labels = clusterer.fit_predict(norm)
    return labels.astype(int)


def _fallback_diarize(
    audio: np.ndarray,
    sample_rate: int,
    num_speakers: int = -1,
) -> List[Dict[str, Any]]:
    """VAD + MFCC clustering fallback.

    Pass ``num_speakers=-1`` to let the clusterer auto-detect the number of
    distinct voices (using silhouette score).
    """
    segments = _silero_vad_segments(audio, sample_rate)
    if not segments:
        return []
    feats = _speaker_embeddings(audio, sample_rate, segments)
    labels = _cluster_embeddings(feats, num_speakers=num_speakers)
    out: List[Dict[str, Any]] = []
    for (ts, te), lab in zip(segments, labels):
        out.append({
            "speaker": f"cluster_{int(lab):02d}",
            "start": round(float(ts), 3),
            "end": round(float(te), 3),
        })
    return out


class DiarizeStage:
    """Detect speaker changes with pyannote.audio (or VAD+clustering fallback)."""

    name = "diarize"

    def __init__(
        self,
        workdir: Path,
        hf_token: Optional[str] = None,
        model_id: str = "pyannote/speaker-diarization-3.1",
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        device: str = "cuda",
        fallback_num_speakers: int = 2,
    ):
        self.workdir = Path(workdir)
        self.hf_token = hf_token
        self.model_id = model_id
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.device = device
        self.fallback_num_speakers = fallback_num_speakers

    def outputs(self) -> List[Path]:
        return [self.workdir / "segments.json"]

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        stage_banner(LOG, 2, 11, "Pyannote Diarization")
        speech_path = Path(context["speech_path"])
        if not speech_path.exists():
            raise FileNotFoundError(speech_path)
        out_path = self.workdir / "segments.json"

        waveform, sr = read_wav(speech_path)
        if waveform.shape[0] > 1:
            mono = waveform.mean(axis=0, keepdims=True)
        else:
            mono = waveform

        if _have_pyannote_auth(self.hf_token):
            try:
                pipeline = _make_pipeline(self.model_id, self.hf_token)
                if pipeline is not None:
                    import torch

                    pipeline.to(torch.device(self.device))
                    audio_in = {
                        "waveform": torch.from_numpy(mono).float(),
                        "sample_rate": sr,
                    }
                    kwargs: Dict[str, Any] = {}
                    if self.min_speakers is not None:
                        kwargs["min_speakers"] = self.min_speakers
                    if self.max_speakers is not None:
                        kwargs["max_speakers"] = self.max_speakers
                    diarization = pipeline(audio_in, **kwargs)
                    raw_segments = _annotation_to_segments(diarization)
                    merged_segments = _merge_tiny_speakers(raw_segments)
                    segments = _relabel_speakers(merged_segments)
                    try:
                        del pipeline
                    except Exception:
                        pass
                    free_vram()
                    speakers = sorted({s["speaker"] for s in segments})
                    LOG.info(f"Pyannote: {len(speakers)} speakers / {len(segments)} segments")
                    out_path.write_text(json.dumps(segments, indent=2, ensure_ascii=False))
                    return {
                        "segments_path": str(out_path),
                        "speakers": speakers,
                        "num_segments": len(segments),
                    }
            except Exception as exc:  # noqa: BLE001
                LOG.warning(f"pyannote diarization failed ({exc}); using VAD+clustering fallback")
                free_vram()
        else:
            LOG.info("No HF_TOKEN; using VAD + MFCC clustering diarization")

        # If the user did not pin min/max speakers, let the clusterer
        # auto-detect.  Otherwise respect the user-provided range.
        if self.max_speakers is None and self.min_speakers is None:
            target = -1
        else:
            target = self.max_speakers or self.fallback_num_speakers
        raw_segments = _fallback_diarize(mono, sr, num_speakers=target)
        merged_segments = _merge_tiny_speakers(raw_segments)
        segments = _relabel_speakers(merged_segments)
        speakers = sorted({s["speaker"] for s in segments})
        LOG.info(f"Fallback diarization: {len(speakers)} speakers / {len(segments)} segments")
        out_path.write_text(json.dumps(segments, indent=2, ensure_ascii=False))
        log_vram(LOG)
        return {
            "segments_path": str(out_path),
            "speakers": speakers,
            "num_segments": len(segments),
        }
