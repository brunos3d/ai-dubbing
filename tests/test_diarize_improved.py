"""Unit tests for the new improved-diarization helpers.

These tests cover the pure-Python pieces of the redesign that don't
require a real pyannote model:

* `_knee_select` (knee/elbow detection on a dendrogram)
* `_select_k` (multi-metric k selection)
* `_reassign_segments_by_majority` (chunk->segment vote)
* `_maybe_merge_similar` (cosine-similarity gated merge)

The integration with the real pyannote pipeline is exercised by the
end-to-end validation in `PYANNOTE_DEBUG_REPORT.md`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Make `src` importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# We import the module under test lazily inside the test bodies so a
# missing optional dep (pyannote) doesn't break the test collection.


def _two_speaker_embeddings(seed: int = 0) -> np.ndarray:
    """Return deterministic embeddings with two well-separated clusters.

    Within-cluster noise is high (scale=0.4) and cluster centroids
    are 1.5 units apart.  This produces a clean two-cluster
    dendrogram: lots of within-cluster merges at moderate distance,
    then ONE big jump to merge the two clusters.
    """
    rng = np.random.default_rng(seed)
    dim = 256
    a = rng.normal(loc=+1.5, scale=0.4, size=(30, dim))
    b = rng.normal(loc=-1.5, scale=0.4, size=(30, dim))
    return np.vstack([a, b])


def _three_speaker_embeddings(seed: int = 0) -> np.ndarray:
    """Return deterministic embeddings with three well-separated clusters.

    Centroids placed far apart in 256-D space; noise is high (scale=0.4)
    so the dendrogram has a clear 3-cluster structure.
    """
    rng = np.random.default_rng(seed)
    dim = 256
    a = rng.normal(loc=+2.0, scale=0.4, size=(20, dim))
    b = rng.normal(loc=-2.0, scale=0.4, size=(20, dim))
    centroid_c = np.zeros(dim)
    centroid_c[: dim // 2] = -2.5
    centroid_c[dim // 2 :] = +2.5
    c = rng.normal(loc=centroid_c, scale=0.4, size=(15, dim))
    return np.vstack([a, b, c])


def test_select_k_picks_3_on_three_separated_clusters() -> None:
    """The multi-metric selector (silhouette + DBI + CH) finds 3 in clean data.

    This is the more reliable detector than the dendrogram knee for
    data with well-separated clusters.
    """
    from src.stages.diarize import _select_k

    embs = _three_speaker_embeddings()
    k_star, trace = _select_k(embs, pyannote_count=2, k_min=2, k_max=5, metric="cosine")
    assert k_star == 3, f"expected k_star=3, got {k_star}; trace={trace}"
    assert isinstance(trace, list)
    assert len(trace) >= 2
    for row in trace:
        assert "k" in row
        assert "composite" in row
        assert "silhouette" in row


def test_select_k_respects_min_speakers_floor() -> None:
    """Even if the metrics prefer a smaller k, min_speakers must hold."""
    from src.stages.diarize import _select_k

    embs = _two_speaker_embeddings()
    k_star, _ = _select_k(embs, pyannote_count=2, k_min=3, k_max=5, metric="cosine")
    assert k_star >= 3, f"min_speakers floor not respected: k_star={k_star}"


def test_select_k_respects_max_speakers_ceiling() -> None:
    from src.stages.diarize import _select_k

    embs = _three_speaker_embeddings()
    k_star, _ = _select_k(embs, pyannote_count=3, k_min=2, k_max=3, metric="cosine")
    assert k_star <= 3, f"max_speakers ceiling not respected: k_star={k_star}"


def test_reassign_segments_majority_vote_assigns_dominant_cluster() -> None:
    from src.stages.diarize import _reassign_segments_by_majority

    # 10 chunks: 6 are cluster A, 4 are cluster B
    chunk_labels = np.array([0] * 6 + [1] * 4)
    # Two segments: first segment contains 5 A-chunks, second contains 1 A + 4 B
    # Both segments should land in cluster A by majority if A is dominant globally,
    # but majority is local so segment 2 should land in cluster B.
    segments = [(0.0, 0.6, "SPEAKER_X"), (0.6, 1.0, "SPEAKER_X")]
    chunk_starts = np.linspace(0.0, 0.9, 10)  # chunk mid = start + 0.05
    chunk_ends = chunk_starts + 0.1

    new_segments, low_conf = _reassign_segments_by_majority(
        segments=segments,
        chunk_starts=chunk_starts,
        chunk_ends=chunk_ends,
        chunk_labels=chunk_labels,
        unknown_label="UNK",
    )
    # Segment 1 covers 0.0-0.6 -> midpoints 0.05, 0.15, 0.25, 0.35, 0.45 (5 chunks) all A
    # Segment 2 covers 0.6-1.0 -> midpoints 0.55, 0.65, 0.75, 0.85 (4 chunks) all B
    # Actually with linspace 0.0 to 0.9 = 10 chunks: midpoints 0.05..0.95 step 0.1
    # Segment 1 (0.0-0.6) has midpoints 0.05..0.55 (chunks 0..5) all A
    # Segment 2 (0.6-1.0) has midpoints 0.65..0.95 (chunks 6..9) all B
    speakers = [s["speaker"] for s in new_segments]
    assert speakers[0] == "speaker_01", f"expected segment 0 -> speaker_01, got {speakers[0]}"
    assert speakers[1] == "speaker_02", f"expected segment 1 -> speaker_02, got {speakers[1]}"
    assert low_conf == 0, f"expected 0 low-confidence, got {low_conf}"


def test_reassign_segments_low_confidence_when_split() -> None:
    from src.stages.diarize import _reassign_segments_by_majority

    # 10 chunks: 5 are cluster A, 5 are cluster B
    chunk_labels = np.array([0, 1] * 5)
    # One segment contains 3 A + 2 B (50/50 split -> low confidence)
    segments = [(0.0, 1.0, "SPEAKER_X")]
    chunk_starts = np.linspace(0.0, 0.9, 10)
    chunk_ends = chunk_starts + 0.1

    new_segments, low_conf = _reassign_segments_by_majority(
        segments=segments,
        chunk_starts=chunk_starts,
        chunk_ends=chunk_ends,
        chunk_labels=chunk_labels,
        unknown_label="UNK",
    )
    assert low_conf == 1, f"expected 1 low-confidence segment (50/50), got {low_conf}"


def test_maybe_merge_similar_collapses_close_speakers() -> None:
    from src.stages.diarize import _maybe_merge_similar

    # Two speakers with IDENTICAL centroids.  Tiny one should collapse
    # into the larger because the cosine distance is ~0.
    centroid_a = np.array([1.0, 0.0, 0.0])
    centroid_b = np.array([1.0, 0.0, 0.0])
    segments = [
        {"speaker": "speaker_01", "start": 0.0, "end": 10.0},
        {"speaker": "speaker_02", "start": 10.0, "end": 11.0},  # 1.0s
    ]
    # 10 chunks at label 0, 2 chunks at label 1
    chunk_labels = np.array([0] * 10 + [1] * 2)
    # chunk_centroids is the per-CLUSTER centroid table (n_clusters, D)
    chunk_centroids = np.stack([centroid_a, centroid_b])  # (2, 3)

    out = _maybe_merge_similar(
        segments=segments,
        chunk_labels=chunk_labels,
        chunk_centroids=chunk_centroids,
        merge_threshold=0.30,
        min_dur=2.0,
    )
    speakers = {s["speaker"] for s in out}
    assert speakers == {"speaker_01"}, f"expected only speaker_01 after merge, got {speakers}"


def test_maybe_merge_similar_keeps_dissimilar_speakers() -> None:
    from src.stages.diarize import _maybe_merge_similar

    centroid_a = np.array([1.0, 0.0, 0.0])
    centroid_b = np.array([-1.0, 0.0, 0.0])  # opposite -> cos_dist = 2.0
    segments = [
        {"speaker": "speaker_01", "start": 0.0, "end": 10.0},
        {"speaker": "speaker_02", "start": 10.0, "end": 11.0},  # tiny
    ]
    chunk_labels = np.array([0] * 10 + [1] * 2)
    chunk_centroids = np.stack([centroid_a, centroid_b])  # (2, 3)

    out = _maybe_merge_similar(
        segments=segments,
        chunk_labels=chunk_labels,
        chunk_centroids=chunk_centroids,
        merge_threshold=0.30,
        min_dur=2.0,
    )
    speakers = {s["speaker"] for s in out}
    assert speakers == {"speaker_01", "speaker_02"}, (
        f"expected both speakers preserved, got {speakers}"
    )


def test_chunk_to_label_remap_is_contiguous_from_zero() -> None:
    """The re-clustering labels should be remapped to 0..N-1 in order of appearance."""
    from src.stages.diarize import _relabel_clusters

    labels = np.array([5, 5, 5, 7, 7, 9])
    remapped = _relabel_clusters(labels)
    assert set(remapped.tolist()) == {0, 1, 2}
    # Same cluster ID should map to the same new ID everywhere
    assert remapped[0] == remapped[1] == remapped[2]
    assert remapped[3] == remapped[4]
    assert remapped[5] != remapped[0]


def test_choose_k_margin_rejects_weak_extra_cluster() -> None:
    """A marginal composite gain over the pyannote prior must not add a speaker.

    Evidence (Elon/Colbert workspace): pyannote found 2 speakers, the
    re-clusterer's composite rose only 0.259 -> 0.341 (gap 0.082) at k=3 — a
    spurious split from audience/overlap chunks. The margin guard keeps k=2.
    """
    from src.stages.diarize import _choose_k_with_margin

    trace = [
        {"k": 2, "composite": 0.259},
        {"k": 3, "composite": 0.341},
        {"k": 4, "composite": 0.321},
    ]
    assert _choose_k_with_margin(trace, base_k=2, margin=0.15) == 2


def test_choose_k_margin_keeps_clear_extra_cluster() -> None:
    """A large composite gain (a genuine speaker) is honoured."""
    from src.stages.diarize import _choose_k_with_margin

    trace = [
        {"k": 2, "composite": 0.66},
        {"k": 3, "composite": 0.899},  # +0.239 — a real third voice
        {"k": 4, "composite": 0.61},
    ]
    assert _choose_k_with_margin(trace, base_k=2, margin=0.15) == 3


def test_choose_k_margin_never_below_floor() -> None:
    from src.stages.diarize import _choose_k_with_margin

    trace = [{"k": 3, "composite": 0.5}, {"k": 4, "composite": 0.52}]
    # best is k=4 but only +0.02 over the floor -> stay at the floor (3).
    assert _choose_k_with_margin(trace, base_k=3, margin=0.15) == 3
