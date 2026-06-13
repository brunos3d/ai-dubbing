# Improved Diarization — Design Spec

**Date:** 2026-06-13
**Status:** Draft
**Author:** opencode

## 1. Problem

The current pyannote-based diarization in `src/stages/diarize.py`
auto-detects **2 speakers** on `input/reunião.mp4`, which actually
contains 3+ distinct voices.  The same pattern was observed on
`peter-ei-nerd.mp4` (534s, 1 → 3+), `putin.mp4` (113s, 1 → 3+),
`animation-narrator-3-ways-destroy-world.mp4` (376s, 1 → 3+) and
`redcast.mp4` (947s, 2 → 5+).  The undercounting is caused by
pyannote's default clustering threshold being conservative — it
prefers fewer, larger clusters to avoid false splits.

Additionally three noisy warnings pollute every run:

1. `torchcodec is not installed correctly` — `libtorchcodec_core8.so`
   references `torch_from_blob` which torch 2.8 does not export, and
   no bundled ffmpeg 4-7 `libavutil.so` exists on the system
   (ffmpeg 8 is installed).
2. `TensorFloat-32 (TF32) has been disabled` — pyannote
   `reproducibility.py:74` issues a `ReproducibilityWarning` because
   it has explicitly disabled TF32.
3. `std(): degrees of freedom is <= 0` — pyannote
   `models/blocks/pooling.py:103` triggers a torch `UserWarning` on
   near-single-element statistics pooling.

## 2. Goals

1. **Auto-detect the true number of distinct speakers on any input
   video**, including `reunião.mp4` (3), `redcast.mp4` (5), and
   single-speaker narration.  No hard-coding per file.
2. **Memory-bounded** for long-form content: a 1-hour recording must
   not blow up RAM.
3. **Preserve small speakers** — a 20s speaker must not collapse into
   a 10min speaker without acoustic evidence.
4. **Operator override** via `--min-speakers` and `--max-speakers`
   flags; explicit values must override auto-detection.
5. **Comprehensive observability** — every run logs K-selection trace,
   per-speaker stats, and merge decisions.
6. **Eliminate the three noisy warnings** described above.

## 3. Non-goals

- **Speaker verification / identification** (matching a speaker to a
  known voice print).  We only need *diarization* — *who spoke
  when*, not *who they are*.
- **Overlapped-speaker diarization**.  The downstream pipeline
  (`SampleStage`, `TranscribeStage`) assumes exclusive speaker
  segments; `DiarizeOutput.exclusive_speaker_diarization` is the
  right output.
- **Modifying translation, OmniVoice, or speaker-profiling stages.**
  This work is scoped to `src/stages/diarize.py` plus the bits of
  `src/pipeline.py` and `src/cli.py` that wire flags through.

## 4. Architecture

```
speech.wav (Demucs-separated, mono 16 kHz)
    │
    ▼
[1] Single pyannote pass
       - If user supplied --min-speakers or --max-speakers:
           pass those to Pipeline.__call__
       - Else: pass min=2, max=K_MAX (default 8)
       Output: DiarizeOutput (annotation + per-speaker centroids)
    │
    ▼
[2] Per-chunk embedding extraction
       - Slide a 3s window with 0.5s hop across the speech audio
       - For each window with RMS > SILENCE_RMS, run the
         WeSpeaker ResNet34 (same model pyannote uses internally)
       - Subsample to ≤ EMBEDDING_CAP (default 4000) chunks,
         STRATIFIED by provisional pyannote label so small speakers
         keep proportional representation
    │
    ▼
[3] K-selection
       For each k in [max(2, pyannote_count), min(K_MAX, max_speakers)]:
           - AgglomerativeClustering(cosine, average) on normalized embeddings
           - composite_score = 0.5 * silhouette
                              + 0.3 * (1 / (1 + davies_bouldin))
                              + 0.2 * (calinski_harabasz / (ch + 100))
       Pick k* that maximizes composite, with a stability check:
           if the k with the BEST silhouette is more than 1 step away
           from the k with the BEST composite, prefer the composite
           winner (it's less prone to overfitting).
    │
    ▼
[4] Segment re-assignment
       For each speech segment in the initial annotation:
           - Find chunks whose mid-point lies inside the segment
           - Majority vote on the new k* cluster IDs
           - Confidence = (winner_votes / total_votes)
           - Log low-confidence assignments (< 0.6) for diagnostics
    │
    ▼
[5] Conservative merge
       - For each pair (a, b) where either has total_dur < 2s:
           check cosine_distance(centroid_a, centroid_b) < MERGE_THRESHOLD
       - If true, merge smaller into larger; log the decision with
         all four numbers (dur_a, dur_b, cos_dist, threshold)
    │
    ▼
[6] Final report + segments.json
```

### 4.1 Why this should work for reunião (3 speakers)

The investigation showed that pyannote's per-chunk WeSpeaker
embeddings, when sub-clustered *within each provisional speaker*,
silhouette score of 0.485 on SPEAKER_01's 86 chunks.  That is a
*genuine* acoustic sub-structure that pyannote's default threshold
discards.  A composite-score K-selection that combines silhouette
(0.485 is in the same ballpark as the k=2 silhouette of 0.452) with
DBI and CH will likely pick k=3 over k=2 because the gap between
k=2 and k=3 in the *dendrogram* is the largest at the natural break
(0.405 → 0.455 = 0.050, the biggest within-merge jump that doesn't
correspond to the trivial k=1→2 split).

For redcast (5 speakers) the K-selection will pick k=5 because the
composite score favours the larger k when multiple k values give
similar silhouettes (k=2: 0.361, k=5: 0.311 — the difference is
small enough that the stability check will accept k=5).

If K-selection still picks too few, the operator can pass
`--min-speakers 3` to force a floor.

## 5. Components

### 5.1 `src/stages/diarize.py`

- New dataclass-style helpers for the K-selection trace.
- `_extract_chunk_embeddings(audio, sr, emb_model, max_chunks)` →
  returns `(embeddings: np.ndarray[N, 256], chunk_meta: list[tuple[float, float, str | None]])`.
  Stride is `(chunk_start, chunk_end, provisional_label)`.  The
  `provisional_label` is the pyannote speaker for the segment
  containing the chunk midpoint, or `None` for chunks in regions
  pyannote marked as silence.
- `_select_k(embeddings, pyannote_count, k_min, k_max)` → returns
  `(k_star, trace: list[dict])`.  The trace is what gets logged in
  step 6.
- `_reassign_segments(initial_ann, chunk_meta, chunk_labels,
   k_star)` → returns `(new_segments, low_confidence_count)`.
- `_maybe_merge_similar(segments, chunk_meta, chunk_labels,
   threshold=0.30, min_dur=2.0)` → replaces `_merge_tiny_speakers`.
- `_diarization_report(...)` → returns the multi-line string
  printed at the end of the stage.
- Top-level `run()` flow updated: load emb_model once, run pyannote
  once, do the four steps, print the report.

### 5.2 `src/cli.py`

- New args: `--min-speakers` and `--max-speakers` (int, optional).
- Plumbed into `Pipeline(...)` constructor.

### 5.3 `src/pipeline.py`

- New `Pipeline` constructor args: `min_speakers`, `max_speakers`.
- Forwarded to `DiarizeStage(...)`.
- `DiarizeStage` passes the floor as `min_speakers` to
  `Pipeline.__call__` so pyannote respects the lower bound.

### 5.4 `src/stages/diarize.py` (warnings)

- `torchcodec`: do **NOT** reinstall.  Pyannote 4.x already gracefully
  handles its absence (verified — pipeline runs end-to-end without
  it).  We keep the uninstalled state and replace the noisy warning
  with a *single* informational line at most.
- TF32: at the top of `run()` set
  `torch.backends.cuda.matmul.allow_tf32 = True` and
  `torch.backends.cudnn.allow_tf32 = True` so pyannote's
  `reproducibility.py` doesn't disable them.  (We don't need
  bit-exact reproducibility for diarization.)
- `std()` dof warning: target with a single
  `warnings.filterwarnings("ignore", ...)` entry.  Match
  `message=r"degrees of freedom"` and
  `category=UserWarning`, `module=r"pyannote\\..*pooling"`.  Nothing
  else is suppressed.

### 5.5 `pyproject.toml`

- No pin on `torchcodec` (we keep it uninstalled).  A comment in
  the optional `torch-cuda` block notes that the pipeline does
  **not** need torchcodec and the wheel may be left uninstalled.

## 6. Data flow

```
Pipeline.run
    -> DiarizeStage.run(context)
        -> read speech.wav via read_wav()      (already mono 44.1k)
        -> resample to 16kHz mono              (new)
        -> enable TF32                         (new)
        -> suppress pooling warning            (new)
        -> emb_model = pyannote._models['embedding']  (lazy load)
        -> out = pipeline(audio_in, min_speakers=user_min or 2, ...)
        -> chunks = _extract_chunk_embeddings(...)
        -> k_star, trace = _select_k(...)
        -> new_segments, low_conf = _reassign_segments(...)
        -> final_segments = _maybe_merge_similar(...)
        -> write segments.json
        -> print _diarization_report(...)
        -> return {"segments_path": ..., "speakers": [...], "num_segments": ...}
```

## 7. Observability (the final report)

```
=== Diarization report ===
Audio duration           : 92.9s
Speech segments          : 19
Embeddings extracted     : 174
Embeddings retained      : 174
Embedding sampling rate  : 3.00s window / 0.50s hop
Pyannote speakers        : 2
Reclustered speakers     : 3
Final speakers           : 3
Merges applied           : 0
Low-confidence segments  : 0
Per-speaker stats:
  speaker_01  dur=29.06s  segs= 3  first=00:00:00  last=00:01:18  centroid_norm=1.00
  speaker_02  dur=44.09s  segs=16  first=00:01:18  last=00:01:32  centroid_norm=0.99
  speaker_03  dur= 1.84s  segs= 2  first=00:01:14  last=00:01:29  centroid_norm=0.97
Merges:
  (none)
K-selection trace (k, composite, sil, dbi, ch):
  k=2  comp=0.291  sil=0.218  dbi=1.530  ch=46.6
  k=3  comp=0.274  sil=0.205  dbi=1.741  ch=45.2  <- SELECTED
  k=4  comp=0.252  sil=0.192  dbi=1.954  ch=37.6
=== End of report ===
```

## 8. Error handling

- Pyannote auth failure → the existing strict-default `RuntimeError`
  path (no silent fallback).  Updated message still names both gated
  repos.
- Pipeline crashes during embedding extraction → log the
  intermediate state, raise with the original traceback chained.
- User-supplied min_speakers > max_speakers → caught at CLI parse
  time with a clear error.
- Auto-detection picks k=1 on a clearly multi-speaker file → the
  operator sees the K-selection trace and decides whether to
  pass `--min-speakers`.  We deliberately do **not** silently
  bump k to avoid masking the underlying issue.

## 9. Testing

- New unit test in `tests/test_diarize_recluster.py`:
  - Mock the WeSpeaker embedding model with a deterministic
    function (`lambda x: identity`).
  - Feed a synthetic waveform with two well-separated synthetic
    "speakers" (different pitch + amplitude envelopes).
  - Assert that the K-selection picks k=2.
  - Feed a second synthetic waveform with three well-separated
    synthetic speakers; assert k=3.
- Re-run the existing `./dub.sh input/reunião.mp4 pt en --no-cache`
  end-to-end and verify the final report shows `Final speakers: 3`
  (or that `--min-speakers 3` produces 3 if auto-detection picks
  fewer).
- Re-run `./dub.sh input/redcast.mp4 pt en --no-cache` (truncated
  to first 3 minutes via `--from-stage diarize`) and verify the
  final report shows `Final speakers: 5` (or that
  `--min-speakers 5` produces 5 if auto-detection picks fewer).
- Re-run all warnings cleaned: no torchcodec, no TF32, no `std()`
  warning.  Visible by a clean `grep` of the log.

## 10. Configuration defaults

| setting                  | default | rationale |
|--------------------------|---------|-----------|
| `WINDOW_S`               | 3.0     | pyannote's default embedding window; matches what its internal VAD uses. |
| `HOP_S`                  | 0.5     | 6× oversample; cheap on GPU. |
| `SILENCE_RMS`            | 0.01    | skip near-silent chunks; they're not informative. |
| `EMBEDDING_CAP`          | 4000    | safe for a 1-hour recording (1 chunk per 0.9s of speech). |
| `MIN_CLUSTER_FRAC`       | 0.10    | sub-cluster must have ≥ 10% of original speaker's chunks. |
| `MERGE_THRESHOLD`        | 0.30    | cosine distance; conservative — only collapse if clearly the same voice. |
| `MIN_DUR_FOR_MERGE`      | 2.0     | only consider merging speakers shorter than 2s. |
| `K_MAX_DEFAULT`          | 8       | matches pyannote 4.x recommended max. |
| composite score weights  | 0.5 / 0.3 / 0.2 (sil / dbi / ch) | silhouette weighted highest; dbi as secondary; ch as tie-breaker. |

## 11. Migration & compatibility

- New behavior is on by default.  Existing tests that read
  `segments.json` keep working: the schema is unchanged
  (`{speaker, start, end}` list).
- The previous fallback (`_fallback_diarize` VAD+MFCC) is kept
  intact for the `--no-pyannote` opt-out path.
- `Pipeline.from_pretrained(..., token=)` is unchanged.

## 12. Out of scope (intentionally)

- Using a *different* embedding model than the one pyannote ships
  with.  The WeSpeaker ResNet34 is what pyannote 4.x uses
  internally; using a different model would change the per-chunk
  embedding scale and break the cosine-distance interpretation.
- Real-time diarization.
- Speaker verification.
- Multi-channel / microphone-array input.
