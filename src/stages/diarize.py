"""Stage 3 - Speaker diarization.

Primary: pyannote.audio (requires ``HF_TOKEN`` to access the gated models).
Fallback: VAD + MFCC clustering using Silero VAD, which is fully open.

The HF_TOKEN must belong to a user who has accepted the gating terms for
``pyannote/segmentation-3.0`` *and* ``pyannote/speaker-diarization-3.1``;
see :mod:`scripts.test_pyannote_auth` for a standalone probe.
"""
from __future__ import annotations

import json
import os
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.audio import read_wav
from ..utils.logging import get_logger, stage_banner
from ..utils.vram import free_vram, log_vram

LOG = get_logger("ai-dubbing.diarize")

# ---- Warning hygiene (set at import time so they apply before
# pyannote's first use) -----------------------------------------------
#
# Note: Python's warnings module matches the ``message`` filter via
# ``re.match``, which only matches at the *start* of the message.
# The pyannote warnings below all begin with a leading newline, so
# the patterns include ``\n`` at the start.
#
# Also note: pyannote.audio adds ~22 of its own warning filters at
# import time, putting them at the FRONT of the filter list.  The
# standard ``warnings.filterwarnings`` call prepends but pyannote's
# later additions clobber our position.  We solve this by *also*
# re-applying our filters AFTER importing pyannote (and any module
# that may transitively import it).

_FILTER_WARNINGS = [
    # The "torchcodec is not installed correctly" warning is emitted
    # at pyannote.audio import time.  The message starts with a
    # leading newline because of pyannote's showwarning formatting.
    dict(
        action="ignore",
        message=r"\ntorchcodec is not installed correctly",
        category=UserWarning,
    ),
    # The "TensorFloat-32 (TF32) has been disabled"
    # ReproducibilityWarning is emitted by
    # pyannote.audio.utils.reproducibility.
    dict(
        action="ignore",
        message=r"TensorFloat-32",
        category=UserWarning,
    ),
    # The "std(): degrees of freedom is <= 0" UserWarning is raised
    # by pyannote.audio.models.blocks.pooling on tiny single-element
    # statistics pooling.
    dict(
        action="ignore",
        message=r"std\(\): degrees of freedom",
        category=UserWarning,
    ),
]


def _apply_warning_filters() -> None:
    """Install our warning filters at the FRONT of the filter list.

    Idempotent: if a filter is already at the front, the next call
    is a no-op.  We can't easily detect that, so we just always
    prepend - duplicate entries don't cause warnings to slip through
    (Python's filter list is "first match wins", and an additional
    identical entry further back is harmless).
    """
    for f in _FILTER_WARNINGS:
        warnings.filterwarnings(**f)


_apply_warning_filters()


def _resolve_hf_token(hf_token: Optional[str]) -> Optional[str]:
    """Return the first non-empty HF_TOKEN source we can find."""
    if hf_token:
        return hf_token
    env_tok = os.environ.get("HF_TOKEN")
    if env_tok:
        return env_tok
    return None


def _log_token_diagnostics(hf_token: Optional[str]) -> None:
    """Emit non-secret diagnostics about how the token was resolved."""
    resolved = _resolve_hf_token(hf_token)
    if resolved:
        masked = f"{resolved[:3]}...{resolved[-3:]} (len={len(resolved)})"
        LOG.info(f"HF_TOKEN loaded: True ({masked})")
    else:
        LOG.warning("HF_TOKEN loaded: False (no token in args or environment)")
    LOG.info("Pyannote auth mode: token= keyword (pyannote.audio >= 4.0)")


def _have_pyannote_auth(hf_token: Optional[str]) -> bool:
    return _resolve_hf_token(hf_token) is not None


def _make_pipeline(model_id: str, hf_token: Optional[str]):
    """Load a pyannote pipeline, preferring the modern ``token=`` kwarg.

    pyannote.audio 4.x removed ``use_auth_token=``; the 3.x name is still
    accepted by some releases.  We try the modern name first, then fall
    back for the 3.x line, and finally pass nothing so huggingface_hub
    can pick up the token from the ``HF_TOKEN`` environment variable.
    """
    from pyannote.audio import Pipeline

    token = _resolve_hf_token(hf_token)
    try:
        return Pipeline.from_pretrained(model_id, token=token)
    except TypeError:
        # pyannote.audio 3.x - the keyword was ``use_auth_token``.
        return Pipeline.from_pretrained(model_id, use_auth_token=token)


def _annotation_to_segments(annotation, min_duration: float = 0.3) -> List[Dict[str, Any]]:
    """Convert pyannote diarization output to a list of {speaker, start, end} dicts.

    pyannote.audio 4.x returns a ``DiarizeOutput`` dataclass with an
    ``exclusive_speaker_diarization`` attribute that is an
    ``pyannote.core.annotation.Annotation``.  Earlier versions returned
    the ``Annotation`` directly.  This helper accepts both.
    """
    # 4.x returns a DiarizeOutput; pull out the exclusive (no-overlap) diarization
    # which is what downstream stages expect.
    if hasattr(annotation, "exclusive_speaker_diarization"):
        annotation = annotation.exclusive_speaker_diarization
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
# Improved-diarization helpers
# -----------------------------

# Constants used by the new K-selection / merge logic.
DEFAULT_WINDOW_S: float = 3.0
DEFAULT_HOP_S: float = 0.5
DEFAULT_SILENCE_RMS: float = 0.01
DEFAULT_EMBEDDING_CAP: int = 4000
DEFAULT_MERGE_THRESHOLD: float = 0.30
DEFAULT_MIN_DUR_FOR_MERGE: float = 2.0
DEFAULT_K_MAX: int = 8
LOW_CONFIDENCE_THRESHOLD: float = 0.6


def _knee_select(
    embeddings: np.ndarray,
    k_min: int = 2,
    k_max: int = 8,
    metric: str = "cosine",
) -> Tuple[int, List[Tuple[int, float, float]]]:
    """Pick k by finding the "knee" of the dendrogram's gap sequence.

    The dendrogram merges single chunks into clusters; the distance of
    each merge is the inter-cluster distance at that point.  Going
    from k clusters to (k-1) clusters costs a "gap" of
    ``merge_dist[k-1] - merge_dist[k-2]``.

    For well-clustered data the gap sequence is monotonically
    *decreasing* (or at least mostly).  The "knee" is the largest k
    such that the gap is still substantial, i.e. at least
    ``THRESHOLD_FRAC`` of the *largest* gap.

    Returns ``(k_star, gaps)`` where ``gaps`` is a list of
    ``(k, merge_dist, gap)`` tuples - one per k considered.

    Note: this helper is *weak* on its own.  It is most useful as a
    tie-breaker after the multi-metric ``_select_k``.  The full
    re-clusterer uses both.
    """
    from scipy.cluster.hierarchy import linkage

    if embeddings.shape[0] < k_min + 1:
        return k_min, []

    if metric == "cosine":
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        embs = embeddings / norms
    else:
        embs = embeddings

    Z = linkage(embs, method="average", metric=metric)
    merge_dists = Z[:, 2]
    diffs = np.diff(merge_dists)

    gaps: List[Tuple[int, float, float]] = []
    for k in range(k_min, min(k_max, len(merge_dists)) + 1):
        idx = len(merge_dists) - k
        if idx <= 0 or idx > len(diffs):
            continue
        gaps.append((k, float(merge_dists[idx]), float(diffs[idx - 1])))

    if not gaps:
        return k_min, []
    if len(gaps) == 1:
        return gaps[0][0], gaps

    # Largest k whose gap is at least 5% of the maximum gap.
    # The 5% threshold keeps the algorithm sensitive to smaller
    # natural splits while ignoring the long tail of within-cluster
    # merges.
    max_gap = max(g[2] for g in gaps)
    threshold = 0.05 * max_gap

    chosen_k = gaps[0][0]
    for k, _, g in gaps:
        if g >= threshold:
            chosen_k = k
    return chosen_k, gaps


def _select_k(
    embeddings: np.ndarray,
    pyannote_count: int,
    k_min: int = 2,
    k_max: int = 8,
    metric: str = "cosine",
) -> Tuple[int, List[Dict[str, Any]]]:
    """Multi-metric K-selection for the re-clustering stage.

    Combines silhouette (cosine), Davies-Bouldin index and
    Calinski-Harabasz score into a single composite score, then
    selects the k that maximises it *subject to*:

    * k is in [k_min, k_max];
    * k is at least the pyannote provisional count (we never reduce).

    Returns ``(k_star, trace)`` where ``trace`` is a list of dicts
    keyed by k and includes all the underlying metrics for logging.
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )
    from sklearn.preprocessing import normalize

    k_min = max(k_min, pyannote_count, 2)
    k_max = max(k_max, k_min)

    # unit-normalise for cosine metrics
    if metric == "cosine":
        embs = normalize(embeddings, norm="l2")
    else:
        embs = embeddings

    if embs.shape[0] < k_min + 1:
        return k_min, []

    trace: List[Dict[str, Any]] = []
    best: Tuple[int, float, float, float, float] = (k_min, -1e9, 0.0, 0.0, 0.0)
    for k in range(k_min, min(k_max, embs.shape[0] - 1) + 1):
        labels = AgglomerativeClustering(
            n_clusters=k, metric=metric, linkage="average"
        ).fit_predict(embs)
        if len(set(labels)) < 2:
            continue
        sil = float(silhouette_score(embs, labels, metric=metric))
        dbi = float(davies_bouldin_score(embs, labels))
        ch = float(calinski_harabasz_score(embs, labels))
        # Normalise to roughly comparable scales
        dbi_n = 1.0 / (1.0 + dbi)        # higher is better; in (0, 1)
        ch_n = ch / (ch + 100.0)         # higher is better; in (0, 1)
        composite = 0.5 * sil + 0.3 * dbi_n + 0.2 * ch_n
        trace.append(
            {"k": k, "composite": composite, "silhouette": sil,
             "davies_bouldin": dbi, "calinski_harabasz": ch}
        )
        if composite > best[1]:
            best = (k, composite, sil, dbi, ch)

    # If the silhouette-best k disagrees with the composite-best by
    # more than 1, prefer the composite winner (less prone to
    # overfitting on a single metric).  In practice this is a no-op
    # when the metrics agree.
    if trace and best[0] != k_min:
        # find silhouette-best
        sil_best_k = max(trace, key=lambda r: r["silhouette"])["k"]
        if abs(sil_best_k - best[0]) > 1:
            LOG.debug(
                f"silhouette preferred k={sil_best_k}, composite preferred "
                f"k={best[0]} - going with composite"
            )

    return best[0], trace


def _relabel_clusters(labels: np.ndarray) -> np.ndarray:
    """Remap cluster IDs to 0..N-1 in order of first appearance."""
    mapping: Dict[int, int] = {}
    out = np.empty_like(labels)
    next_id = 0
    for i, lab in enumerate(labels):
        lab_i = int(lab)
        if lab_i not in mapping:
            mapping[lab_i] = next_id
            next_id += 1
        out[i] = mapping[lab_i]
    return out


def _reassign_segments_by_majority(
    segments: List[Tuple[float, float, str]],
    chunk_starts: np.ndarray,
    chunk_ends: np.ndarray,
    chunk_labels: np.ndarray,
    unknown_label: str = "UNK",
) -> Tuple[List[Dict[str, Any]], int]:
    """Re-assign each segment to a new cluster ID by majority vote of the
    chunks whose mid-points fall inside the segment.

    Returns ``(new_segments, low_confidence_count)`` where
    ``low_confidence_count`` is the number of segments whose winning
    cluster had less than ``LOW_CONFIDENCE_THRESHOLD`` of the vote.
    """
    new_segments: List[Dict[str, Any]] = []
    low_conf = 0
    chunk_mids = (chunk_starts + chunk_ends) / 2.0
    for idx, (start, end, _orig_speaker) in enumerate(segments):
        mask = (chunk_mids >= start) & (chunk_mids < end)
        if not mask.any():
            new_segments.append(
                {"speaker": unknown_label, "start": round(start, 3),
                 "end": round(end, 3), "confidence": 0.0}
            )
            continue
        votes = chunk_labels[mask]
        # majority vote
        unique, counts = np.unique(votes, return_counts=True)
        winner = int(unique[np.argmax(counts)])
        confidence = float(counts.max() / counts.sum())
        new_segments.append(
            {"speaker": f"speaker_{winner + 1:02d}", "start": round(start, 3),
             "end": round(end, 3), "confidence": round(confidence, 3)}
        )
        if confidence < LOW_CONFIDENCE_THRESHOLD:
            low_conf += 1
    return new_segments, low_conf


def _maybe_merge_similar(
    segments: List[Dict[str, Any]],
    chunk_labels: np.ndarray,
    chunk_centroids: np.ndarray,
    merge_threshold: float = DEFAULT_MERGE_THRESHOLD,
    min_dur: float = DEFAULT_MIN_DUR_FOR_MERGE,
) -> List[Dict[str, Any]]:
    """Conservative merge: only collapse two speakers if BOTH are short
    AND their centroids are very close in cosine distance.

    The previous `_merge_tiny_speakers` collapsed small speakers into
    the *dominant* speaker on duration/segment-count alone, which made
    short participants disappear.  This version is gated on actual
    acoustic similarity: small speakers stay distinct unless the
    embeddings say they're the same voice.

    ``chunk_labels`` and ``chunk_centroids`` must be parallel arrays
    whose index i refers to the same chunk.  ``segments`` carries the
    human-readable speaker name of each speech segment.  The merge
    candidate pool is the set of speakers whose total duration is
    below ``min_dur``.

    Note: ``chunk_centroids`` is the *per-cluster* centroid table
    (shape ``(n_clusters, D)``) returned by ``_compute_chunk_centroids``.
    To get a per-speaker centroid, we average the centroids of the
    cluster labels that the segments of that speaker touched.
    """
    if not segments:
        return segments
    if chunk_labels.size == 0 or chunk_centroids.size == 0:
        return segments

    # Per-speaker duration
    speaker_dur: Dict[str, float] = {}
    for seg in segments:
        spk = seg["speaker"]
        speaker_dur[spk] = speaker_dur.get(spk, 0.0) + (seg["end"] - seg["start"])

    if len(speaker_dur) < 2:
        return segments

    # Per-speaker set of cluster labels that its chunks belong to
    # (since segments are reassigned by majority vote over chunks).
    speaker_clusters: Dict[str, set] = {}
    for spk in speaker_dur:
        # The reassign step stores speakers as "speaker_NN" - extract the
        # compact int label.
        try:
            lab = int(spk.rsplit("_", 1)[-1]) - 1
        except (ValueError, IndexError):
            continue
        # Find which chunk labels this speaker's chunks had
        # (chunks that majority-vote to this speaker).
        # We use a different approach: the cluster centroids are indexed
        # by the int label; we map speaker_NN -> the int N-1 -> that
        # centroid row.
        speaker_clusters[spk] = {lab}

    # Centroid per speaker
    speaker_centroid: Dict[str, np.ndarray] = {}
    for spk, cluster_set in speaker_clusters.items():
        rows = [chunk_centroids[c] for c in cluster_set if c < chunk_centroids.shape[0]]
        if not rows:
            continue
        c = np.mean(np.stack(rows), axis=0)
        n = float(np.linalg.norm(c))
        if n > 1e-9:
            c = c / n
        speaker_centroid[spk] = c

    # Find merge candidates
    small_speakers = [s for s, d in speaker_dur.items() if d < min_dur]
    if not small_speakers:
        return segments

    merge_map: Dict[str, str] = {}
    for spk in small_speakers:
        if spk not in speaker_centroid:
            continue
        c_a = speaker_centroid[spk]
        best_target: Optional[str] = None
        best_dist = float("inf")
        for other, c_b in speaker_centroid.items():
            if other == spk or other in merge_map:
                continue
            d = 1.0 - float(np.dot(c_a, c_b))
            if d < best_dist:
                best_dist = d
                best_target = other
        if best_target is not None and best_dist < merge_threshold:
            merge_map[spk] = best_target
            LOG.info(
                f"Merging speaker {spk} -> {best_target} "
                f"(dur={speaker_dur[spk]:.2f}s, cos_dist={best_dist:.3f}, "
                f"threshold={merge_threshold})"
            )

    if not merge_map:
        return segments

    out = []
    for seg in segments:
        new = dict(seg)
        if new["speaker"] in merge_map:
            new["speaker"] = merge_map[new["speaker"]]
        new.pop("confidence", None)
        out.append(new)
    return out


def _extract_chunk_embeddings(
    audio: np.ndarray,
    sample_rate: int,
    emb_model: Any,
    pyannote_annotation: Any,
    device: Any = None,
    window_s: float = DEFAULT_WINDOW_S,
    hop_s: float = DEFAULT_HOP_S,
    silence_rms: float = DEFAULT_SILENCE_RMS,
    embedding_cap: int = DEFAULT_EMBEDDING_CAP,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, List[int]]]:
    """Run the embedding model on sliding windows over the audio.

    Returns ``(embeddings, chunk_starts, chunk_labels, per_speaker_indices)``
    where:

    * ``embeddings``: ``(N, D)`` float32 array
    * ``chunk_starts``: ``(N,)`` float array of chunk start times in seconds
    * ``chunk_labels``: ``(N,)`` int array of the provisional pyannote
      speaker label for the segment containing the chunk midpoint, or
      ``-1`` for chunks in pyannote-silent regions
    * ``per_speaker_indices``: dict from provisional label to list of
      indices into the other arrays
    """
    import torch

    if audio.ndim > 1:
        mono = audio.mean(axis=0)
    else:
        mono = audio

    target_sr = 16000
    if sample_rate != target_sr:
        try:
            import librosa  # type: ignore
            mono = librosa.resample(mono.astype("float32"), orig_sr=sample_rate, target_sr=target_sr)
        except Exception:
            # No resample - skip (embedding model expects 16 kHz)
            return np.empty((0, 0), dtype=np.float32), np.empty(0), np.empty(0, dtype=int), {}
        sr = target_sr
    else:
        sr = sample_rate

    # Build chunk mid-points and the provisional speaker label
    # for each one (the pyannote label of the segment that contains it).
    chunk_records: List[Tuple[float, float, int]] = []
    for t in np.arange(0.0, len(mono) / sr - window_s + 1e-9, hop_s):
        s0 = int(t * sr)
        s1 = int((t + window_s) * sr)
        if s1 > len(mono):
            break
        chunk = mono[s0:s1]
        if np.sqrt(np.mean(chunk ** 2)) < silence_rms:
            continue
        mid = t + window_s / 2.0
        # Find the provisional speaker (label of the segment containing mid)
        spk_label = -1
        for turn, _, speaker in pyannote_annotation.itertracks(yield_label=True):
            if turn.start <= mid <= turn.end:
                # Map the string label to a stable int
                spk_label = speaker
                break
        chunk_records.append((t, t + window_s, spk_label))

    if not chunk_records:
        return np.empty((0, 0), dtype=np.float32), np.empty(0), np.empty(0, dtype=int), {}

    # Stratified subsample: protect minority speakers.
    # Group chunks by their string provisional label
    by_label: Dict[str, List[int]] = {}
    for i, (_, _, lbl) in enumerate(chunk_records):
        by_label.setdefault(str(lbl), []).append(i)

    if sum(len(v) for v in by_label.values()) > embedding_cap:
        rng = np.random.default_rng(0xC0FFEE)
        # Equal-share budget, but never less than 1 per group that has chunks
        n_groups = max(1, len(by_label))
        per_group_cap = max(1, embedding_cap // n_groups)
        # For temporal spread: within each group, sample evenly across the time axis
        kept: List[int] = []
        for label, indices in by_label.items():
            indices_sorted = sorted(indices, key=lambda i: chunk_records[i][0])
            if len(indices_sorted) <= per_group_cap:
                kept.extend(indices_sorted)
            else:
                # Evenly-spaced picks
                picks = np.linspace(0, len(indices_sorted) - 1, per_group_cap, dtype=int)
                kept.extend(indices_sorted[i] for i in picks)
        # If we're still over budget (one group was huge), randomly drop
        if len(kept) > embedding_cap:
            kept = list(rng.choice(kept, size=embedding_cap, replace=False))
        chunk_records = [chunk_records[i] for i in sorted(kept)]

    # Now run the embedding model in batches
    n_chunks = len(chunk_records)
    emb_dim: Optional[int] = None
    embeddings = np.empty((n_chunks, 0), dtype=np.float32)
    for batch_start in range(0, n_chunks, 16):
        batch = []
        for (cs, ce, _) in chunk_records[batch_start:batch_start + 16]:
            s0 = int(cs * sr)
            s1 = int(ce * sr)
            batch.append(mono[s0:s1].astype("float32"))
        batch_t = torch.from_numpy(np.stack(batch)).unsqueeze(1)  # (B, 1, samples)
        if device is not None:
            batch_t = batch_t.to(device)
        with torch.no_grad():
            out = emb_model(batch_t)
        if isinstance(out, tuple):
            out = out[0]
        emb = out.detach().cpu().numpy().astype("float32")
        embeddings = emb if embeddings.size == 0 else np.vstack([embeddings, emb])
        emb_dim = emb.shape[1]

    chunk_starts = np.array([r[0] for r in chunk_records], dtype=np.float32)
    # Map string labels to compact int IDs, preserving first-appearance order
    label_to_id: Dict[str, int] = {}
    next_id = 0
    chunk_labels = np.empty(n_chunks, dtype=int)
    for i, (_, _, lbl) in enumerate(chunk_records):
        key = str(lbl)
        if key not in label_to_id:
            label_to_id[key] = next_id
            next_id += 1
        chunk_labels[i] = label_to_id[key]

    # Build per_speaker_indices for the temporary diagnostic + reassign
    per_speaker_indices: Dict[str, List[int]] = {
        lbl: [i for i, lab in enumerate(chunk_labels) if lab == idx]
        for lbl, idx in label_to_id.items()
    }
    return embeddings, chunk_starts, chunk_labels, per_speaker_indices


def _stratified_temporal_subsample(
    indices: List[int], chunk_starts: np.ndarray, n_keep: int
) -> List[int]:
    """Pick ``n_keep`` indices spread evenly across the temporal range."""
    if len(indices) <= n_keep:
        return sorted(indices)
    sorted_idx = sorted(indices, key=lambda i: chunk_starts[i])
    picks = np.linspace(0, len(sorted_idx) - 1, n_keep, dtype=int)
    return [sorted_idx[i] for i in picks]


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
    inputs: List[str] = ["media/speech.wav"]
    outputs: List[str] = [
        "diarization/segments.json",
        "diarization/embeddings.npz",
        "diarization/embeddings.meta.json",
        "diarization/metadata.json",
    ]
    editable_outputs: List[str] = []
    derived_outputs: List[str] = ["diarization/embeddings.npz"]
    config_fields: List[str] = [
        "model_id",
        "min_speakers",
        "max_speakers",
        "no_pyannote",
    ]

    def __init__(
        self,
        workdir: Path,
        hf_token: Optional[str] = None,
        model_id: str = "pyannote/speaker-diarization-3.1",
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        device: str = "cuda",
        fallback_num_speakers: int = 2,
        no_pyannote: bool = False,
        subdir: str | None = None,
    ):
        self.workdir = Path(workdir)
        if subdir:
            self.workdir = self.workdir / subdir
        self.hf_token = hf_token
        self.model_id = model_id
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.device = device
        self.fallback_num_speakers = fallback_num_speakers
        # When True, skip pyannote entirely and use the VAD+MFCC fallback.
        # Pyannote is the default diarizer - opting out must be explicit.
        self.no_pyannote = no_pyannote
        self.subdir = subdir

    def output_paths(self) -> List[Path]:
        return [self.workdir / "segments.json"]

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        import warnings
        import torch

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

        # ---- Warning hygiene ----
        # Enable TF32 explicitly: pyannote's reproducibility module
        # turns it off and emits a warning.  We don't need bit-exact
        # reproducibility for diarization, and TF32 gives a small
        # speedup on Ampere+ GPUs.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Re-apply our warning filters in case pyannote re-shuffled
        # the list when it was imported.  Cheap and idempotent.
        _apply_warning_filters()

        _log_token_diagnostics(self.hf_token)

        if self.no_pyannote:
            LOG.info("--no-pyannote set; using VAD + MFCC clustering fallback")
            return self._run_fallback(mono, sr, out_path)

        if not _have_pyannote_auth(self.hf_token):
            raise RuntimeError(
                "HF_TOKEN is not set but pyannote is the default diarizer.  "
                "Either set HF_TOKEN in .env (the token must belong to a "
                "Hugging Face account that has accepted the gating terms "
                "for pyannote/segmentation-3.0 and "
                "pyannote/speaker-diarization-3.1) or re-run with "
                "--no-pyannote to use the VAD+MFCC clustering fallback."
            )

        # ---- Pyannote path ----
        try:
            pipeline = _make_pipeline(self.model_id, self.hf_token)
            if pipeline is None:
                raise RuntimeError(
                    "Pipeline.from_pretrained returned None - the token "
                    "is valid but the user has not been granted "
                    "access to the gated pyannote models.  Visit "
                    "https://huggingface.co/pyannote/segmentation-3.0 "
                    "and https://huggingface.co/pyannote/speaker-diarization-3.1 "
                    "to accept the user conditions."
                )
            pipeline.to(torch.device(self.device))
            audio_in = {
                "waveform": torch.from_numpy(mono).float(),
                "sample_rate": sr,
            }
            # Pass only the *user* min/max to pyannote so it can
            # respect the operator's hard floor/ceiling.  The
            # re-clusterer below will pick a *finer* k within that
            # range.
            kwargs: Dict[str, Any] = {}
            if self.min_speakers is not None:
                kwargs["min_speakers"] = self.min_speakers
            if self.max_speakers is not None:
                kwargs["max_speakers"] = self.max_speakers
            out_obj = pipeline(audio_in, **kwargs)
            initial_ann = (
                out_obj.exclusive_speaker_diarization
                if hasattr(out_obj, "exclusive_speaker_diarization")
                else out_obj
            )
            n_pyannote = len(set(initial_ann.labels()))
            LOG.info(f"Pyannote provisional speakers: {n_pyannote}")
            initial_segments = _annotation_to_segments(out_obj)
            # Drop the diagnostic 'confidence' key if any
            for s in initial_segments:
                s.pop("confidence", None)
            try:
                del pipeline
            except Exception:
                pass
            free_vram()

            # ---- Re-clustering with per-chunk embeddings ----
            LOG.info("Re-clustering speaker embeddings to refine speaker count...")
            emb_model = _get_pyannote_embedding_model(
                self.model_id, self.hf_token, self.device
            )
            embeddings, chunk_starts, chunk_labels_int, _ = _extract_chunk_embeddings(
                mono, sr, emb_model, initial_ann,
                device=torch.device(self.device),
            )
            n_chunks = embeddings.shape[0]
            n_retained = n_chunks
            if n_chunks < 4:
                LOG.warning(
                    f"Only {n_chunks} non-silent chunks found; skipping "
                    f"re-clustering and keeping pyannote's output."
                )
                final_segments = initial_segments
                n_reclustered = n_pyannote
                trace = []
            else:
                # K bounds for re-clusterer: respect user floor,
                # default to pyannote's count if the user didn't pin.
                k_min_user = self.min_speakers if self.min_speakers is not None else n_pyannote
                k_max_user = self.max_speakers if self.max_speakers is not None else DEFAULT_K_MAX
                k_star, trace = _select_k(
                    embeddings,
                    pyannote_count=n_pyannote,
                    k_min=k_min_user,
                    k_max=k_max_user,
                    metric="cosine",
                )
                n_reclustered = k_star
                LOG.info(f"Reclustered speakers: {n_reclustered}")

                # Relabel cluster IDs to 0..N-1
                chunk_labels = _relabel_clusters(chunk_labels_int)

                # Build the per-chunk centroid table for the merge step
                # (centroid is the *mean* of all chunks in a cluster,
                # not the per-chunk embedding).
                chunk_centroids = _compute_chunk_centroids(
                    embeddings, chunk_labels, n_reclustered
                )

                # Map provisional segments to (start, end, label) tuples
                seg_tuples = [
                    (s["start"], s["end"], s["speaker"]) for s in initial_segments
                ]
                # Use the chunk mid-points
                chunk_mids = (chunk_starts + (chunk_starts + DEFAULT_WINDOW_S)) / 2.0
                new_segments, low_conf = _reassign_segments_by_majority(
                    segments=seg_tuples,
                    chunk_starts=chunk_mids - DEFAULT_WINDOW_S / 2.0,
                    chunk_ends=chunk_mids + DEFAULT_WINDOW_S / 2.0,
                    chunk_labels=chunk_labels,
                )
                LOG.info(f"Low-confidence segments: {low_conf}")

                # Conservative merge: only merge tiny speakers whose
                # centroids are actually close.
                final_segments = _maybe_merge_similar(
                    new_segments,
                    chunk_labels,
                    chunk_centroids,
                )

            # Drop diagnostic-only keys before persisting
            for s in final_segments:
                s.pop("confidence", None)
            n_final = len({s["speaker"] for s in final_segments})
            LOG.info(f"Final speakers (after merge): {n_final}")
            LOG.info(f"Detected speakers: {n_final}")
            out_path.write_text(json.dumps(final_segments, indent=2, ensure_ascii=False))

            # ---- Report ----
            self._print_diarization_report(
                mono=mono, sr=sr,
                n_initial_segments=len(initial_segments),
                n_extracted=n_chunks, n_retained=n_retained,
                n_pyannote=n_pyannote, n_reclustered=n_reclustered,
                n_final=n_final, final_segments=final_segments,
                trace=trace, merges=None, low_conf=low_conf if n_chunks >= 4 else 0,
            )
            return {
                "segments_path": str(out_path),
                "speakers": sorted({s["speaker"] for s in final_segments}),
                "num_segments": len(final_segments),
            }
        except Exception as exc:  # noqa: BLE001
            LOG.error(f"pyannote diarization failed: {type(exc).__name__}: {exc}")
            LOG.debug(traceback.format_exc())
            raise RuntimeError(
                "Pyannote diarization failed and no fallback is allowed "
                "(default behaviour).  Re-run with --no-pyannote to opt "
                "out of pyannote and use the VAD+MFCC clustering "
                "fallback, or fix the underlying error above."
            ) from exc

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _silence_torchcodec_warning() -> None:
        """No-op kept for backwards compatibility.

        The torchcodec / TF32 / std() warning filters are set at
        *module import time* (see top of file) so they apply before
        pyannote is imported, not at stage run-time.  This method is
        preserved so any external code calling it still works.
        """
        return None

    def _run_fallback(
        self, mono: np.ndarray, sr: int, out_path: Path
    ) -> Dict[str, Any]:
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

    @staticmethod
    def _print_diarization_report(
        *,
        mono: np.ndarray,
        sr: int,
        n_initial_segments: int,
        n_extracted: int,
        n_retained: int,
        n_pyannote: int,
        n_reclustered: int,
        n_final: int,
        final_segments: List[Dict[str, Any]],
        trace: List[Dict[str, Any]],
        merges: Optional[List[Dict[str, Any]]],
        low_conf: int,
    ) -> None:
        duration = mono.shape[-1] / float(sr) if mono.ndim > 0 else 0.0

        # Per-speaker stats
        speaker_stats: Dict[str, Dict[str, Any]] = {}
        for s in final_segments:
            spk = s["speaker"]
            d = s["end"] - s["start"]
            if spk not in speaker_stats:
                speaker_stats[spk] = {"dur": 0.0, "segs": 0, "first": s["start"], "last": s["end"]}
            speaker_stats[spk]["dur"] += d
            speaker_stats[spk]["segs"] += 1
            speaker_stats[spk]["first"] = min(speaker_stats[spk]["first"], s["start"])
            speaker_stats[spk]["last"] = max(speaker_stats[spk]["last"], s["end"])

        def _hms(t: float) -> str:
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = t - h * 3600 - m * 60
            return f"{h:02d}:{m:02d}:{s:05.2f}"

        LOG.info("=== Diarization report ===")
        LOG.info(f"Audio duration           : {duration:.1f}s")
        LOG.info(f"Speech segments          : {n_initial_segments}")
        LOG.info(f"Embeddings extracted     : {n_extracted}")
        LOG.info(f"Embeddings retained      : {n_retained}")
        LOG.info(f"Embedding sampling rate  : {DEFAULT_WINDOW_S:.2f}s window / {DEFAULT_HOP_S:.2f}s hop")
        LOG.info(f"Pyannote speakers        : {n_pyannote}")
        LOG.info(f"Reclustered speakers     : {n_reclustered}")
        LOG.info(f"Final speakers           : {n_final}")
        LOG.info(f"Low-confidence segments  : {low_conf}")
        if merges:
            LOG.info(f"Merges applied           : {len(merges)}")
            for m in merges:
                LOG.info(
                    f"  merged {m['src']} -> {m['dst']}  "
                    f"dur_src={m['dur_src']:.2f}s  cos_dist={m['cos_dist']:.3f}  "
                    f"threshold={m['threshold']:.3f}"
                )
        else:
            LOG.info("Merges applied           : 0")
        LOG.info("Per-speaker stats:")
        for spk, st in sorted(speaker_stats.items()):
            LOG.info(
                f"  {spk}  dur={st['dur']:6.2f}s  segs={st['segs']:3d}  "
                f"first={_hms(st['first'])}  last={_hms(st['last'])}"
            )
        if trace:
            LOG.info("K-selection trace (k, composite, sil, dbi, ch):")
            for row in trace:
                marker = "  <- SELECTED" if row["k"] == n_reclustered else ""
                LOG.info(
                    f"  k={row['k']:2d}  comp={row['composite']:.3f}  "
                    f"sil={row['silhouette']:.3f}  "
                    f"dbi={row['davies_bouldin']:.3f}  "
                    f"ch={row['calinski_harabasz']:.1f}{marker}"
                )
        LOG.info("=== End of report ===")


def _get_pyannote_embedding_model(model_id: str, hf_token: Optional[str], device_str: str):
    """Instantiate and return the WeSpeaker embedding model used by pyannote.

    We pull it from the same model_id so we share the auth path and
    the same model revision.  Loading is lazy inside pyannote, so
    forcing the load before we ask for the embedding keeps timing
    predictable.
    """
    import torch
    from pyannote.audio.pipelines.utils import get_model

    pipeline = _make_pipeline(model_id, hf_token)
    emb_ref = pipeline.embedding
    emb_model = get_model(emb_ref, token=_resolve_hf_token(hf_token))
    emb_model.to(torch.device(device_str))
    emb_model.eval()
    # The pipeline itself is no longer needed
    try:
        del pipeline
    except Exception:
        pass
    return emb_model


def _compute_chunk_centroids(
    embeddings: np.ndarray, chunk_labels: np.ndarray, n_clusters: int
) -> np.ndarray:
    """Compute the per-cluster centroid (mean of its chunks' embeddings).

    Returns ``(n_clusters, D)`` array.  Empty clusters get a zero row
    (they shouldn't exist if K-selection worked, but defend anyway).
    """
    if embeddings.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    dim = embeddings.shape[1]
    out = np.zeros((n_clusters, dim), dtype=np.float32)
    for c in range(n_clusters):
        mask = chunk_labels == c
        if mask.sum() > 0:
            out[c] = embeddings[mask].mean(axis=0)
    return out
