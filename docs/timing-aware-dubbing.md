# Timing-Aware Dubbing, Timeline & Prosody

This guide covers the quality features added in the 2026-06 evolution cycle
(spec: `docs/superpowers/specs/2026-06-16-timing-aware-timeline-evolution.md`).
Everything below integrates with the Workspace architecture — it participates
in invalidation, exposes metadata, and stays user-editable.

## What changed, in one paragraph

Translation used to be timing-blind: the pipeline translated literally, then
stretched the synthesized audio to fit. Now the translate stage **estimates
how long each candidate translation will take to speak** and picks the one
that best fits the original slot, so stretching is a last resort. The
delivery of each line (tempo, loudness, pauses) is measured from the source
audio and carried into synthesis and mixing. Editing one translated line
**re-renders only that line**. A new `timeline.json` gives every segment one
structured, inspectable record. The final mix **ducks the background under
the dialogue** and lightly **matches the room** so the voice sits in the
scene.

## Timing-aware translation

For each segment the translate stage:

1. collects multiple translation **candidates** (one per provider);
2. estimates each candidate's **spoken duration** (a language-aware syllable
   model: syllables ÷ speaking-rate + pause budget);
3. scores candidates by a weighted blend of **duration fit**, **rate
   naturalness** (0.90–1.15× band), **fidelity**, and **glossary
   compliance**; and
4. selects the best fit.

Each segment in `translation/translated_transcript.json` gains a `timing`
block, and the stage writes a human-readable `translation/timing_report.json`:

```json
{
  "summary": {"n_segments": 120, "mean_score": 0.86, "segments_needing_stretch": 4},
  "segments": [
    {"slot_duration_s": 3.10, "estimated_duration_s": 3.04,
     "speaking_rate_factor": 0.98, "score": 0.91, "selected_text": "…"}
  ]
}
```

**Tuning:** weights and the natural-rate band live in `src/timing/score.py`;
per-language speaking rates in `src/timing/duration.py`.

## The Timeline (`timeline.json`)

A canonical per-segment document, rebuilt from the existing artifacts and
written to `<workspace>/timeline.json`. Each segment carries speaker, span,
source/target text, glossary hits, timing, prosody, generation/reuse state,
review flags, and a `render_key`. It is a **derived** artifact (never triggers
invalidation) and **coexists** with the legacy JSON — stages still read those.

```bash
./dub.sh workspace inspect <id>          # prints a Timeline summary
./dub.sh workspace show <id> timeline.json
```

## Segment-level re-rendering

The generate stage computes a stable `render_key` per segment (target text +
speaker + voice-profile hash + model + slot duration + prosody signature). On
re-run it **reuses** any segment whose key is unchanged and re-synthesizes
only the rest. If nothing changed, the TTS model is never even loaded.

Practical effect: fix one line in `translated_transcript.json`, run
`./dub.sh generate <id>`, and only that line is re-synthesized.

## Prosody transfer

Per segment, the pipeline measures from the source speech (`speech.wav`):
energy, voiced/pause ratio, speaking rate, and (guarded) pitch. It then:

* maps the segment's relative speaking rate → the TTS **`speed`** so fast/slow
  delivery transfers;
* maps relative loudness → a per-segment **gain** applied at reconstruction,
  preserving emphasis across a turn.

Descriptors are written to `generated_segments/prosody.json` and onto the
Timeline, and they fold into the `render_key` (a prosody change re-renders).

## Acoustic matching

The final mix now:

* **ducks** the music/effects bed under the dialogue using a sidechain
  compressor keyed on the new dialogue (so it follows the *dubbed* timing),
  falling back to the static mix if unavailable; and
* applies a **light, capped room reverb** to the dry synthetic voice, sized
  from a reverberance estimate of the original speech stem (never doubling
  reverb on already-wet input).

Both are `MixStage` config (`ducking`, `room_match`, default on) and are
recorded in the manifest, so toggling them re-runs the mix via the DAG.

## Performance & footprint

All new analysis is **CPU-only, no new heavy dependencies, no network**
(duration ≈ 76k estimates/s; candidate scoring ≈ 24k/s; prosody ≈ 13 ms per
3 s segment — negligible next to synthesis). CPU fallbacks remain intact.
