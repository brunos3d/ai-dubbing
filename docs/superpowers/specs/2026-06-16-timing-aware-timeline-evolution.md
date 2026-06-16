# Timing-Aware Dubbing & Timeline Foundation — Design Spec

**Date:** 2026-06-16
**Status:** Draft
**Author:** Claude (Opus 4.8)

## 1. Problem

The platform's architectural audit (`FINAL_REPORT.md`) concluded that the
biggest remaining quality limits are architectural, not model-related:

1. **Translation is timing-unaware.** `translate.py` runs literal MT
   (Google → MyMemory) with no knowledge of how long the result will
   take to speak. A faithful translation that cannot be spoken in the
   available slot is still chosen, and the damage is pushed downstream.
2. **Timing is solved *after* translation.** `align.py` force-fits the
   synthesized audio to the original duration with `ffmpeg atempo`, and
   `reconstruct.py` time-stretches *again* past 1.4×. This is the
   "translate first, desperately stretch" anti-pattern; it rushes or
   drags speech and stacks two lossy resampling passes.
3. **Prosody / emotion is discarded.** Nothing reads *how* a line was
   delivered. OmniVoice clones timbre from one reference and reads the
   translation flatly.
4. **Invalidation is stage-level only.** Editing one translated line
   re-runs `generate → align → reconstruct → mix → video` for *every*
   segment, not just the changed one.
5. **Architecture is stage-oriented, not Timeline-oriented.** Stages
   pass opaque JSON files through a stringly-typed `context` dict.
   There is no canonical, queryable, per-segment representation.
6. **`Pipeline` and `WorkspacePipeline` coexist**, duplicating context
   hydration and breeding bugs (visible in the commit log).
7. **Acoustic matching is minimal.** Dry synthetic voice sits over the
   original background at a static −6 dB with no ducking and no room
   match — the classic "voice over background" tell.
8. **No persistence above the job** (Speaker Registry / Translation
   Memory). Out of scope for this cycle (see §3) but noted.

## 2. Goals

This is an **evolutionary** cycle. No rewrites. Every feature integrates
with the existing Workspace Architecture, Manifest DAG, and invalidation
logic, exposes metadata, and stays user-editable.

1. **Consolidate** the orchestrator: one execution path, validators that
   match reality, no redundant model loads / stretches / hashing.
2. **Timing-aware translation:** generate translation *candidates*,
   estimate spoken duration *before* synthesis, rank candidates by a
   composite timing score, and *prefer a better-fitting translation over
   stretching audio*. Stretching becomes a last resort.
3. **Timeline v1:** introduce a canonical, structured, per-segment
   document (`timeline.json`) that coexists with current artifacts and
   breaks nothing.
4. **Segment-level intelligence:** per-segment render hashing so
   `generate` re-synthesizes only the segments that actually changed.
5. **Prosody preservation:** extract speaking-rate / energy / pause /
   pitch descriptors from `speech.wav` and condition synthesis + mixing
   on them, using only libraries already present (librosa, numpy).
6. **Acoustic matching:** dynamic ducking (sidechain) and light room
   matching so the dub sits *in* the scene, not on top of it.

## 3. Non-goals (deferred to a later cycle)

- Persistent Speaker Registry / Translation Memory / cross-episode
  linking. (Timeline v1 is designed so these can hang off it later.)
- A REST API or UI. (Timeline v1 is the data model they will use.)
- A learned prosody latent or neural duration predictor. (Heuristic
  duration + descriptor transfer only — local-first, no new heavy deps.)
- Lip-sync / visual dubbing (any tier).
- Replacing OmniVoice or the translation provider.
- Multimodal (video) diarization.

## 4. Architecture overview

Target processing flow (the feedback edge is the new part):

```
extract → separate → diarize → samples → transcribe
                                              │
                                              ▼
                                        translate
                                   (candidates + duration
                                    estimate + timing score)
                                              │  selected candidate
                                              ▼  + timing plan
                                          generate ──── prosody conditioning
                                       (segment-level skip)
                                              │
                                              ▼
                                           align  (minimal correction only)
                                              │
                                              ▼
                                       reconstruct (per-segment gain match)
                                              │
                                              ▼
                                           mix  (dynamic ducking + room match)
                                              │
                                              ▼
                                           video
```

New/changed modules:

| Module | Role |
|---|---|
| `src/timing/duration.py` *(new)* | Language-aware spoken-duration estimator (syllable model). |
| `src/timing/score.py` *(new)* | Candidate timing score + selection. |
| `src/stages/translate.py` *(changed)* | Multi-candidate translation + timing-aware selection + report. |
| `src/workspace/timeline.py` *(new)* | `Timeline` / `Segment` dataclasses, build-from-artifacts, persist. |
| `src/stages/generate.py` *(changed)* | Per-segment render hash → skip unchanged; prosody conditioning. |
| `src/timing/prosody.py` *(new)* | Per-segment prosody descriptors from `speech.wav`. |
| `src/stages/align.py` *(changed)* | Minimal correction; single stretch authority. |
| `src/stages/reconstruct.py` *(changed)* | Per-segment gain match; drop the second stretch. |
| `src/stages/mix.py` *(changed)* | Dynamic ducking (sidechain) + optional room match. |
| `src/pipeline.py` *(changed)* | Reduced to a thin compatibility shim. |
| `src/workspace/validate.py` *(changed)* | Accept the artifacts actually produced. |

The Timeline is **derived**: it is rebuilt from the existing artifacts
(diarization, transcript, translation, generate manifest) and persisted
as `timeline.json`. It is added to `derived_paths` so it never triggers
invalidation. Stages do not yet *read* from it (they keep reading the
existing JSON files) — Timeline v1 is a read-model and metadata surface.
This keeps backwards compatibility absolute while establishing the
canonical representation the platform will grow into.

---

## PHASE 1 — Architectural consolidation

### 1A. Single orchestrator

**Decision: `WorkspacePipeline` becomes the single orchestrator;
`Pipeline` becomes a thin compatibility shim.**

Rationale: the workspace path already owns manifest, DAG, atomic writes,
and is the most complete. The CLI `run` command currently builds the
legacy `Pipeline` directly — that is the duplication.

Changes:
- `cli.py` `run` routes through `WorkspacePipeline.prepare()` +
  `.generate()` (one-shot users still get identical outputs *and* a
  workspace transparently — this is what the README already claims).
- `Pipeline` keeps `pipeline_config_dict()` and `STAGES` (used by tests
  and for back-compat) but `run()` delegates to `WorkspacePipeline`. The
  duplicated `_rehydrate_from_disk` / `_hydrate_context` bodies are
  removed. Importing `Pipeline` keeps working; no external break.
- Legacy temp-cache (`utils/cache.py`) stays for `dub.sh cache`, but is
  no longer on the `run` happy path.

Back-compat guard: `tests/test_pipeline_config_dict.py` must still pass.

### 1B. Validators match reality

**Decision: fix the validators to accept what the stages emit** (lower
blast radius than rewriting every stage's output schema and re-hashing
every artifact downstream).

Today stages emit **bare JSON lists** (`diarize` → `[{speaker,start,end}]`;
`transcribe`/`translate` → `[{...}]`) but `validate.py` requires
`{$schema_version, segments, speakers, id, ...}`. The validators are
made *shape-tolerant*: each `validate_*` accepts **either** the legacy
bare list **or** the v1 dict, validating the per-segment fields that are
actually present. Errors stay errors only for genuinely malformed data.

### 1C. Eliminate redundant work

- **Reference re-transcription:** `generate.py` reloads `whisper-tiny`
  and re-transcribes every speaker reference even though `samples.py`
  already wrote `transcript.txt` / `transcript_text`. Fix: use the
  on-disk transcript; only fall back to whisper if it is empty.
- **Double time-stretch:** `align.py` stretches to target; then
  `reconstruct.py` stretches again past 1.4×. Fix: `align` is the single
  stretch authority; `reconstruct` only places audio (no second stretch).
- **Double manifest save:** `pipeline.py` saves the manifest twice in a
  row — remove the duplicate. Remove dead `... and False:` branch.
- Verify no other repeated full-file hashing within a single
  `compute_stale_set` pass (it already memoizes per path; confirm).

---

## PHASE 2 — Timing-aware dubbing

### 2.1 Duration estimation (`src/timing/duration.py`)

A dependency-free, language-aware estimator:

```
estimate_duration(text: str, lang: str) -> DurationEstimate
```

Model: **syllables ÷ speaking-rate + pause budget.**

- **Syllable count:** per-language vowel-group heuristic. For Latin-script
  languages, count maximal runs of vowels (incl. accented) as one nucleus;
  apply small per-language corrections (e.g. Portuguese/Spanish diphthong
  handling, English silent-final-`e`). Fallback: `chars / avg_chars_per_syllable`.
- **Speaking rate (syllables/sec)** per language, from published
  speech-tempo literature, e.g. `en≈4.4, es≈5.4, pt≈4.9, fr≈5.0, de≈4.3,
  it≈5.0, ja≈5.7, default≈4.6`. Stored in a table, overridable.
- **Pause budget:** add fixed pauses for sentence/clause punctuation
  (`. ? ! ; :` → ~0.25 s; `,` → ~0.12 s) capped at a fraction of total.

Returns `DurationEstimate(seconds, syllables, rate_used, pause_s, method)`.
Pure function, fully unit-testable, CPU-only.

### 2.2 Translation candidates (`translate.py`)

`TranslationBackend.translate_candidates(text, src, tgt) -> list[str]`
(default impl: distinct outputs from each configured provider — Google,
MyMemory — deduplicated; the abstract `translate()` stays for back-compat
and returns `candidates[0]`).

For each segment we collect candidates from all providers plus, when a
glossary is active, the entity-protected variant. Candidates that fail
are skipped; if *all* fail, the protected original is the sole candidate
(current behaviour preserved).

### 2.3 Timing score & selection (`src/timing/score.py`)

For each candidate compute:

- `duration_fit` — `1 − clamp(|est − slot| / slot, 0, 1)`.
- `rate_penalty` — penalty proportional to how far the implied speaking
  rate factor (`est / slot`) is from the natural band `[0.9, 1.15]`.
- `fidelity` — proxy from source/target length ratio (penalize extreme
  expansion/compression vs the language-pair's expected ratio).
- `glossary_compliance` — `1.0` if every active glossary term survives in
  the candidate, else a penalty per missing term.

```
score = 0.45*duration_fit + 0.20*(1−rate_penalty)
      + 0.20*fidelity + 0.15*glossary_compliance
```

`select_candidate(segment, candidates, slot, lang, glossary)` returns the
best candidate plus the full per-candidate score breakdown. Weights live
in one constants block so they are tunable.

### 2.4 Prefer translation over stretching

Selection happens in `translate`. The chosen candidate is written to
`translated_transcript.json` (as today) with **new** per-segment timing
metadata (see schema below). Because the selected candidate already fits
the slot, `align.py`'s stretch is rarely triggered and, when it is, the
required factor is small (inaudible). Stretching is now genuinely last
resort.

### 2.5 Timing report

`translation/timing_report.json` + a logged table:

```json
{
  "$schema_version": 1,
  "language_pair": "pt->en",
  "segments": [
    {
      "id": "seg_0001",
      "slot_duration_s": 3.10,
      "selected": {"text": "...", "estimated_duration_s": 3.04, "score": 0.91},
      "candidates": [
        {"text": "...", "estimated_duration_s": 3.04, "score": 0.91, "backend": "google"},
        {"text": "...", "estimated_duration_s": 3.62, "score": 0.74, "backend": "mymemory"}
      ]
    }
  ],
  "summary": {"mean_score": 0.86, "segments_needing_stretch": 4, "n_segments": 120}
}
```

### 2.6 New per-segment fields in `translated_transcript.json`

Additive only (downstream readers ignore unknown keys):

```json
{
  "speaker": "speaker_01",
  "start": 0.52, "end": 3.62,
  "source_text": "Olá, tudo bem?",
  "text": "Hello, how are you?",
  "timing": {
    "slot_duration_s": 3.10,
    "estimated_duration_s": 3.04,
    "speaking_rate_factor": 0.98,
    "score": 0.91,
    "n_candidates": 2
  }
}
```

---

## PHASE 3 — Timeline v1 (`src/workspace/timeline.py`)

A canonical, structured, per-segment read-model that **coexists** with
existing artifacts and is rebuilt from them.

```python
@dataclass
class TimelineSegment:
    id: str                       # "seg_0001"
    speaker: str
    start: float
    end: float
    source_text: str
    target_text: str
    glossary_hits: list[str]
    timing: dict                  # slot/estimated/rate/score (phase 2)
    prosody: dict                 # rate/energy/pause/pitch (phase 5)
    generation: dict              # render_key, render_path, generated_duration, method
    review: dict                  # confidence, low_confidence, locked
    render_key: str               # phase-4 segment hash

@dataclass
class Timeline:
    schema_version: int
    workspace_id: str
    language_pair: str
    segments: list[TimelineSegment]
    # build_from_workspace(root) -> Timeline
    # save(path) / load(path)
```

- Persisted at `<workspace>/timeline.json`.
- Built after `translate` (text + timing) and refreshed after
  `generate`/`align` (generation + render state).
- Added to manifest `derived_paths` → never triggers invalidation.
- `dub.sh workspace show <id> timeline` prints its path;
  `workspace inspect` gains a one-line Timeline summary.
- **Editable-in-principle but read-model for now:** stages keep reading
  the legacy JSON. A later cycle flips the dependency.

Backwards compatibility: if `timeline.json` is absent (old workspace),
everything still works; it is generated on the next `generate`.

---

## PHASE 4 — Segment-level intelligence

### 4.1 Render key

Per segment, a stable hash of everything that affects its synthesized
audio:

```
render_key = sha256(
    target_text || speaker || voice_profile_hash ||
    target_language || tts_model_id ||
    round(slot_duration, 3) || prosody_signature
)[:16]
```

`voice_profile_hash` is the speaker's `primary.wav`+`primary.txt` hash
(already specified in the workspace spec §12). `prosody_signature` is a
compact rounding of the prosody descriptors so a prosody edit re-renders.

### 4.2 Segment-level skip in `generate`

`generate.py` loads the *previous* generate manifest (and/or
`timeline.json`). For each segment:

- compute `render_key`;
- if a previous segment file exists with the same `render_key` and the
  wav is on disk → **reuse it** (copy/keep, mark `reused: true`);
- else synthesize.

This delivers segment-level regeneration **inside** the existing
stage/DAG model without changing the workspace invalidation contract:
the `generate` stage still runs, but does near-zero work when only one
line changed. Low risk, fully backwards compatible (absent previous
manifest → synthesize everything, today's behaviour).

The generate report logs `reused N / synthesized M`.

### 4.3 Why not change the DAG to segment granularity now

Changing `compute_stale_set` to operate on segments would alter the
manifest schema and the invalidation contract that has tests pinned to
it (spec §10 table). Doing the skip *inside* `generate` achieves the
user-visible win (edit one line → one re-render) with none of that risk.
The render-key groundwork in the Timeline makes a future DAG-level
change straightforward. **Implement 4.1–4.2; document 4.3 as the path.**

---

## PHASE 5 — Prosody & speech quality (`src/timing/prosody.py`)

Per-segment descriptors extracted from `speech.wav` (already on disk),
using librosa + numpy only:

```
analyze_segment(mono, sr, start, end) -> ProsodyDescriptor(
    speaking_rate_syl_s,   # source syllables / voiced time
    energy_rms,            # mean RMS (relative loudness)
    energy_db,             # dBFS
    pitch_mean_hz,         # librosa.pyin median (voiced)
    pitch_range_semitones, # robust p10..p90 spread
    pause_ratio,           # silence / total within the segment
)
```

Usage (practical, no model changes):

1. **Tempo character → TTS `speed`.** Map the source segment's relative
   speaking rate (vs the speaker's mean) to OmniVoice's `speed` argument
   within a safe band so fast/slow delivery transfers.
2. **Relative loudness → per-segment gain** applied in `reconstruct` so
   emphasis/de-emphasis across a turn is preserved (matched to source
   RMS per segment rather than flat-normalizing every line).
3. **Descriptors recorded in the Timeline** as editable fields and fed
   into the `render_key` (phase 4) and the prosody report.

CPU-only, bounded (operates on short per-segment clips). Pitch extraction
(`pyin`) is gated by a fast guard so very long segments stay cheap.

`prosody/prosody.json` report + Timeline `prosody` block.

---

## PHASE 6 — Acoustic matching (`mix.py`, `reconstruct.py`)

Make the dub sit *in* the scene:

1. **Dynamic ducking (primary win).** Replace the static
   `background_db=-6` with an `ffmpeg sidechaincompress`: the background
   is ducked by the dialogue envelope, so music/ambience drops only while
   someone speaks and returns in the gaps — following the *new* dialogue
   timing, exactly as the reference §3.12 wants. Falls back to the static
   mix if the filter is unavailable.
2. **Light room match (optional, cheap).** Estimate a crude reverb
   amount from the original speech stem (energy-decay / late-to-early
   ratio). If the scene is reverberant, apply a light `ffmpeg aecho`
   matched to the estimate so the dry synthetic voice gains comparable
   space. Conservative defaults; never doubles reverb on already-wet
   input (gated by the estimate).
3. **Voice placement.** Keep dialogue centered and at the reference LUFS;
   ducking + room match do the spatial work.

All parameters recorded in the mix stage `config`/report so they
participate in config-change invalidation and stay inspectable.

---

## Workspace integration (applies to every phase)

- New artifacts (`timeline.json`, `timing_report.json`, `prosody.json`)
  are declared on their producing stage's `outputs`; reports are
  `derived_paths` so they don't cause spurious invalidation, while
  `translated_transcript.json` stays editable as today.
- New `config_fields` (timing weights, speaking-rate band, ducking
  params, room-match toggle) are added to the relevant stages so a config
  change re-runs the right stage via the existing DAG.
- All new per-segment metadata is additive; unknown keys are ignored by
  old readers.

## Engineering constraints

- **No rewrites; evolutionary diffs only.**
- **Backwards compatible:** old workspaces and `dub.sh run`/`prepare`/
  `generate` behave identically; new fields are additive.
- **VRAM-reasonable & CPU-fallback intact:** new work (duration, score,
  prosody, ducking) is CPU-only; no new GPU load. Prosody pitch
  extraction is guarded for long clips.
- **Local-first:** no new network deps; duration/score/prosody are
  offline. Candidates reuse the existing online providers (already the
  status quo for translation).
- **Tests for every capability** (see below).
- **Docs updated**; **a commit per milestone**.

## Testing strategy

New test files (one capability each):

- `tests/test_timing_duration.py` — syllable counts & duration estimates
  across en/pt/es; monotonicity (longer text → longer estimate);
  language-rate ordering; empty/punctuation edge cases.
- `tests/test_timing_score.py` — score ranking picks the better-fitting
  candidate; glossary non-compliance penalized; weights sum sanity.
- `tests/test_translate_candidates.py` — candidate collection &
  selection with a stub backend (no network); report shape.
- `tests/test_timeline.py` — build-from-artifacts on a synthetic
  workspace; round-trip save/load; absent-timeline tolerance.
- `tests/test_segment_render_key.py` — render key stability &
  sensitivity (text/voice/timing/prosody change → key change); skip
  logic reuses unchanged, re-synthesizes changed (stubbed TTS).
- `tests/test_prosody.py` — descriptors on synthetic tones (known
  pitch/energy); pause-ratio on speech+silence fixture.
- `tests/test_mix_ducking.py` — filtergraph construction + fallback when
  the sidechain filter is unavailable (no ffmpeg invocation needed for
  the unit; an integration test gated on ffmpeg presence).
- Consolidation: extend `test_workspace_validate.py` with bare-list
  fixtures; keep `test_pipeline_config_dict.py` green.

Backwards-compat: the full existing suite must stay green at every
milestone.

## Files

**NEW**
- `src/timing/__init__.py`
- `src/timing/duration.py`
- `src/timing/score.py`
- `src/timing/prosody.py`
- `src/workspace/timeline.py`
- `tests/test_timing_duration.py`
- `tests/test_timing_score.py`
- `tests/test_translate_candidates.py`
- `tests/test_timeline.py`
- `tests/test_segment_render_key.py`
- `tests/test_prosody.py`
- `tests/test_mix_ducking.py`
- `docs/timing-aware-dubbing.md`

**MODIFIED**
- `src/stages/translate.py` — candidates + selection + report + timing fields
- `src/stages/generate.py` — reuse ref transcript; prosody `speed`; render-key skip
- `src/stages/align.py` — single stretch authority; minimal correction
- `src/stages/reconstruct.py` — per-segment gain match; drop 2nd stretch
- `src/stages/mix.py` — dynamic ducking + room match
- `src/workspace/pipeline.py` — build/refresh Timeline; wire reports
- `src/workspace/validate.py` — shape-tolerant validators
- `src/pipeline.py` — thin compatibility shim
- `src/cli.py` — `run` routes through WorkspacePipeline; `workspace inspect` Timeline line
- `README.md` / `docs/workspaces.md` — document the new behaviour

**UNCHANGED**
- `src/workspace/{manifest,dag,atomic,content_hash,paths}.py` (DAG contract preserved)
- `src/utils/*` (except where redundant work is removed)

## Milestones (commit per milestone)

1. **M1 — Consolidation** (Phase 1A/1B/1C).
2. **M2 — Timing-aware translation** (Phase 2: duration + score + candidates + report).
3. **M3 — Timeline v1** (Phase 3).
4. **M4 — Segment-level rendering** (Phase 4.1/4.2).
5. **M5 — Prosody** (Phase 5).
6. **M6 — Acoustic matching** (Phase 6).
7. **M7 — Docs + deliverables report.**

## Out of scope (explicitly deferred)

Speaker Registry, Translation Memory, REST API, UI, neural prosody,
lip-sync, multimodal diarization, DAG-level (vs in-stage) segment
invalidation. Timeline v1 is the substrate these will build on.
