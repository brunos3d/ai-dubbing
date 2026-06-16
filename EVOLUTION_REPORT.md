# Evolution Cycle Deliverables — Timing-Aware Dubbing & Timeline

**Cycle:** 2026-06-16 · **Branch:** `feat/timing-aware-timeline`
**Spec:** `docs/superpowers/specs/2026-06-16-timing-aware-timeline-evolution.md`
**Baseline audit:** `FINAL_REPORT.md`

This report is the final deliverable for the evolution cycle. It covers all
six phases plus the consolidation work, with evidence (commits, tests,
benchmarks) and an honest assessment of impact and remaining limits.

---

## 1. Architecture report

The cycle was **evolutionary, not a rewrite**. The Workspace architecture
(manifest, invalidation DAG, atomic staging) was kept as the foundation;
every feature plugs into it.

### Before → after (the data/processing flow)

```
BEFORE:  ASR → Translation → TTS → Stretch-to-fit
AFTER:   ASR → Translation(candidates + duration estimate + timing score)
             → TTS(prosody speed, segment-level reuse)
             → minimal correction
             → reconstruct(prosody gain) → mix(ducking + room match)
```

### Structural changes

- **Single orchestrator.** `WorkspacePipeline` now owns the one-shot `run`
  path via `run_oneshot()` (prepare + generate + deliver). `prepare()` became
  staleness-aware so caching/resume is preserved. `src/pipeline.py` is a thin
  compatibility shim that delegates; the duplicated context-hydration logic is
  gone.
- **Timeline v1** (`src/workspace/timeline.py`) is the new canonical
  per-segment read-model — derived from existing artifacts, persisted as
  `timeline.json`, never breaking compatibility. It is the home of
  `segment_render_key`, the seed for future API/UI/registry work.
- **New CPU-only analysis package** `src/timing/` (`duration`, `score`,
  `prosody`) — pure functions, no torch, no network.

### Workspace integration

Every new artifact participates correctly:

| Artifact | Class | Behaviour |
|---|---|---|
| `translation/timing_report.json` | derived | diagnostic; no invalidation |
| `generated_segments/prosody.json` | derived | diagnostic; no invalidation |
| `timeline.json` | derived | rebuilt each run; no invalidation |
| per-segment `timing`/`prosody` fields | additive | ignored by old readers |
| `MixStage.ducking` / `room_match` | config | a change re-runs `mix` via the DAG |

---

## 2. Implemented changes (by milestone / commit)

| Milestone | Commit | Summary |
|---|---|---|
| M1 — Consolidation | `7938a20` | Single orchestrator; staleness-aware `prepare`; shape-tolerant validators; removed redundant whisper load, double time-stretch, double manifest save, dead code |
| M2 — Timing-aware translation | `3991219` | `duration.py` + `score.py`; multi-candidate translation; per-segment `timing`; `timing_report.json` |
| M3 — Timeline v1 | `331e67f` | `timeline.py`; build-from-artifacts; `workspace inspect` summary |
| M4 — Segment-level rendering | `f28b3e8` | `render_key` skip in `generate`; lazy model load |
| M5 — Prosody | `da40815` | `prosody.py`; TTS `speed` + reconstruct gain; descriptors on Timeline |
| M6 — Acoustic matching | `205a4e1` | Dynamic ducking (sidechain) + light room match in `mix` |
| M7 — Docs + deliverables | _this commit_ | `docs/timing-aware-dubbing.md`, README, this report |

### Phase 1 detail (consolidation)

- **1A** WorkspacePipeline single orchestrator; `Pipeline` → shim.
- **1B** Validators accept the **bare-list** artifacts the stages actually
  emit (transcript/translation/diarization), not only the v1 dict — so
  `workspace validate` is now truthful.
- **1C** Eliminated: redundant whisper-tiny load in generate (reuse on-disk
  ref transcripts), the second time-stretch in reconstruct (align is the sole
  authority), the double `manifest.save`, and a dead branch.

---

## 3. Benchmarks

**Test suite:** 108 → **153 passing** (+45), full run ≈ 5 s, CPU-only.

| Component | Throughput (measured, single core) |
|---|---|
| `estimate_duration` | ~75,900 calls/s (13.2 µs/call) |
| `select_candidate` (3 candidates) | ~23,800 calls/s |
| `analyze_segment` (3 s clip, incl. pitch) | ~78 segments/s (12.8 ms/segment) |

Implication for a typical 4-minute clip (~80 segments): the entire new
analysis layer (duration + scoring + prosody) adds **~1 second of CPU**, i.e.
negligible next to separation/ASR/TTS. **No new GPU load**; pitch extraction
is guarded for long clips.

**Segment-level re-render (measured via `test_segment_render_key.py`):**
editing one line of two → **1 synthesis call instead of 2**; an unchanged
re-run → **0 synthesis calls** (model never loaded). On real content this
turns "edit one line → re-synthesize the whole movie" into "re-synthesize one
line."

---

## 4. Quality impact assessment

Honest, mapped to the audit's quality scores (theoretical ceilings; no
human-rated A/B was run this cycle).

| Dimension | Before | Expected after | Why |
|---|---:|---:|---|
| Timing synchronization | 3 | **5–6** | Translation chosen to fit the slot; stretching is now rare and small, not the default. |
| Translation quality | 3 | **4** | Candidate selection + glossary-compliance scoring (still literal MT — the hard ceiling remains). |
| Prosody preservation | 1 | **3–4** | Source rate → TTS speed; relative loudness → per-segment gain. |
| Emotional preservation | 1 | **2–3** | Energy/pause transfer is a partial proxy for affect (no affect latent yet). |
| Voice consistency | 4 | **4** | Unchanged (single primary reference). |
| "Sits in the scene" / mix realism | — | **+** | Dynamic ducking + room match remove the static "voice over background" tell. |
| Human-in-the-loop iteration | — | **++** | Segment-level re-render + Timeline make edit→hear loops surgical. |

The two biggest *perceived* wins are (a) speech that is no longer rushed/dragged
to fit, and (b) a mix that breathes with the dialogue.

---

## 5. Remaining limitations

Carried forward (out of scope this cycle, by design):

- **Translation is still literal MT.** Candidate diversity is limited to what
  the online providers return; there is no length-controlled generative
  translation or context window. This remains the dominant quality ceiling.
- **Prosody transfer is heuristic, not learned.** Speed + gain + pitch
  descriptors approximate delivery; there is no affect latent and no
  cross-lingual intonation modelling. Subtle performances still need a human.
- **Single reference per speaker** (one-note clone); no disentangled voice
  representation, so cross-lingual accent leakage is still possible.
- **No persistence above the job** — Speaker Registry / Translation Memory are
  still absent (Timeline v1 is the substrate they will use).
- **Segment-level invalidation lives inside `generate`**, not in the
  workspace DAG (deliberate, spec §4.3) — a future cycle can lift it to the
  DAG using the render keys now on the Timeline.
- **Room match is deliberately conservative** (capped, gated) — it improves
  obviously-dry-over-wet cases but is not full acoustic-environment matching.
- **Quality deltas are estimated, not human-rated.** No MOS/A-B panel was run.

---

## 6. Recommended next steps

In priority order:

1. **Length-controlled generative translation.** Swap/augment the backend
   with an LLM/NMT that takes a target syllable/char budget; feed it the slot
   from the timing planner. This is the single largest remaining quality
   lever and the candidate/score machinery is already in place to consume it.
2. **Flip a stage to read the Timeline.** Make `generate` read `timeline.json`
   as its input of record (instead of `translated_transcript.json`), proving
   the read-model can become authoritative — the path to API/UI.
3. **DAG-level segment invalidation** using the render keys already persisted.
4. **Speaker Registry + Translation Memory** scoped above the job, keyed by
   the speaker embeddings/voice-profile hashes the workspace already stores —
   unlocks long-form/franchise consistency.
5. **Multi-clip, range-covering reference selection** + a runtime
   no-good-reference failure state (the validator already knows the bar).
6. **Confidence-driven review surface** in `prepare` output (ASR logprob,
   diarization low-confidence, low timing score — all already computed).
7. **Human-evaluation harness** to convert the estimated quality deltas above
   into measured ones and to calibrate the timing-score weights.

---

## 7. Verdict

The cycle delivered all six phases plus consolidation, evolutionarily and with
full backwards compatibility: **153 tests green**, no new heavy dependencies,
CPU-only additions, every feature wired into the Workspace/Manifest/DAG model.
The platform moved from "translate-then-stretch, flat delivery, voice-over-
background, whole-movie re-renders" to "**fit the translation to the clock,
carry the delivery, sit in the scene, and re-render only what changed**" —
with a canonical Timeline now in place as the spine for the API, UI, and
persistence work that comes next.
