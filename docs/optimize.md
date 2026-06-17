# `optimize.sh` — pipeline self-optimization

`optimize.sh` is the terminal front-end for the automatic pipeline-optimization
subsystem (`src/optimization/`). It repeatedly runs the **same** video through
the dubbing pipeline (OmniVoice backend), measures how closely the generated
output matches the source, and searches for the parameter configuration that
maximises that measured quality — autonomously, resumably, and without filling
the disk.

It is **not** a model trainer. It tunes the existing pipeline's parameters
(speaker-reference length, synthesis speed/fitting, alignment thresholds, …).

---

## The same-language idea

Optimization is driven by a **same-language benchmark** — e.g. English → English
(`-l en` on an English clip). When source and target language are identical the
words are known, so the output can be compared directly against the source:
timing, speaker identity, prosody, continuity and spectral reconstruction all
become measurable. A pipeline tuned on this controllable case carries over to
real multilingual dubbing.

---

## Quick start

```bash
# Infinite, resumable run on the Elon/Colbert clip (Ctrl-C to stop)
./optimize.sh run -m input/elon-musk-might-be-a-super-villain.mp4 -l en

# Bounded run (50 iterations)
./optimize.sh run -m input/elon-musk-might-be-a-super-villain.mp4 -l en -n 50

# Check progress at any time (from another terminal)
./optimize.sh status -m input/elon-musk-might-be-a-super-villain.mp4 -l en

# Show the best configuration found so far
./optimize.sh best   -m input/elon-musk-might-be-a-super-villain.mp4 -l en

# Fast, GPU-free smoke test of the loop (synthetic scorer)
./optimize.sh run -m input/elon-musk-might-be-a-super-villain.mp4 -l en --mock -n 30
```

> **Prerequisites for real runs:** a CUDA GPU, the OmniVoice backend, and a
> valid `HF_TOKEN` in `.env` (same requirements as `dub.sh`). `--mock` needs
> none of these.

---

## Commands

| Command | What it does |
|---|---|
| `run` *(default)* | Run the optimization loop with a clean, live per-iteration display. |
| `status` | Print a one-shot snapshot: best score, score progression, last iteration, current output file. |
| `watch` | Live-tail the raw pipeline log for the run. |
| `best` | Print the best configuration discovered so far (`best.json`). |
| `clean` | Delete this run's optimization history (workspace artifacts are kept). |
| `help` | Print full usage. |

If you omit the command, `run` is assumed.

---

## Flags

All commands accept the run-identifying flags (`-m`, `-l`, `-r`); `run` accepts
the rest.

| Flag | Default | Meaning |
|---|---|---|
| `-m, --media PATH` | *(required)* | Source video/audio file. |
| `-l, --language CODE` | `en` | Same-language benchmark code (source == target). |
| `-n, --iterations N` | `0` | Number of iterations. **`0` (or negative) = infinite** until Ctrl-C. |
| `-r, --run-name NAME` | `<slug>-<lang>-<lang>` | Optimization run id (its history directory). |
| `-s, --seed N` | `1234` | RNG seed (makes a run reproducible). |
| `--to-stage NAME` | `reconstruct` | Last pipeline stage to run each iteration. `reconstruct` is enough to score and skips the expensive mix/video remux. |
| `--epsilon F` | `0.2` | Probability of a random-exploration proposal (vs. a greedy neighbour). |
| `--restart-patience N` | `8` | Random restart after this many non-improving steps. |
| `--interval SEC` | `5` | How often the live display refreshes. |
| `--mock` | off | Use a synthetic scorer — validates the loop with no GPU/pipeline. |
| `--verbose` | off | Also stream the raw pipeline log to the terminal. |
| `-- …` | — | Everything after `--` is forwarded verbatim to `scripts/optimize_pipeline.py`. |

---

## Reading the live output

During `run`, each completed iteration prints a block like:

```
── iter 3    [search]  score 0.94633  NEW BEST   best 0.94633 (iter 3)
   from:generate   118.7s
   timing 0.88  slot 0.99  spkSim 0.99  spkCons 0.99  end 1.00  cont 0.97  pros 0.55  recon 0.99
   ⏳ Voice Generation (omnivoice)
```

* **`iter N`** — iteration index (contiguous, resumable).
* **`[baseline|search|random]`** — how the configuration was proposed:
  * `baseline` — the default config (always evaluated first, as the reference);
  * `search` — a greedy neighbour of the current best (hill-climb);
  * `random` — exploration / restart to escape a local optimum.
* **`score`** — the composite quality score in `[0, 1]` (higher is better).
* **`NEW BEST` / `ok` / `FAILED`** — outcome of the iteration.
* **`best … (iter k)`** — best score so far and where it was found.
* **`from:<stage>`** — the earliest pipeline stage that re-ran (only the stages
  affected by the changed parameters run; everything upstream is reused).
* **`<n>s`** — wall-clock time for the iteration.
* The **metrics line** — the per-component scores (see *Metrics* below).
* **`⏳ …`** — a live indicator of the stage the pipeline is currently in,
  shown between iterations.

The full, verbose pipeline log streams to `<run_dir>/run.log` — view it with
`./optimize.sh watch …` or `--verbose`.

---

## Metrics (the composite score)

Each iteration's score is a weighted average of these components, all in
`[0, 1]`, higher = better. A component that can't be computed is dropped and the
weights renormalise.

| Short | Metric | Measures |
|---|---|---|
| `timing` | `timing_accuracy` | Per-segment final duration vs. the source slot. |
| `slot` | `slot_fit` | Penalises overruns (speech bleeding past its slot). |
| `spkSim` | `speaker_similarity` | Cloned-voice timbre vs. the real speaker reference. |
| `spkCons` | `speaker_consistency` | A speaker sounds like one stable person across the dub. |
| `end` | `ending_quality` | Detects clipped / abruptly-cut segment endings. |
| `cont` | `continuity` | Penalises temporal collisions between consecutive turns. |
| `pros` | `prosodic_similarity` | Source vs. generated energy contour (emphasis preserved). |
| `recon` | `reconstruction_quality` | Global duration + spectral-envelope match vs. source speech. |

Default weights live in `src/optimization/metrics.py::DEFAULT_WEIGHTS`.

---

## Resuming, stopping, and storage

* **Resumable:** state, best config and full history are persisted every
  iteration. Press **Ctrl-C** to stop; re-run the *same command* to continue
  exactly where it left off (the search RNG and best are restored).
* **Bounded disk:** every iteration reuses **one** workspace (the tuned
  parameters are excluded from the workspace-identity hash) and overwrites
  segment audio in place, so the footprint stays roughly constant rather than
  growing per iteration. The optimization run directory itself stores only
  small JSON.

### Files written

Per run, under `~/.local/share/ai-dubbing/optimization/<run-name>/`
(override the parent with `AI_DUBBING_OPT_ROOT`):

| File | Contents |
|---|---|
| `history.jsonl` | One JSON record per iteration (score, parameters, metrics, status, timing). |
| `best.json` | The best record seen so far. |
| `state.json` | Loop state for resume (RNG, stale-step counter, workspace root). |
| `run.log` | The full pipeline log for the most recent `run`. |

The reused dubbing workspace lives under
`~/.local/share/ai-dubbing/workspaces/<workspace-id>/`; its
`output/reconstructed_speech.wav` is the current dub being scored.

---

## Promotion & Defaults Registry

Once a run discovers a configuration that significantly improves quality, you
can promote it to the global defaults registry.

### The Defaults Registry

Pipeline parameters are sourced from a central source of truth:
`config/pipeline.defaults.json` in the project root. When this file exists,
its values override the code-defined defaults for all future `dub.sh` and
`optimize.sh` runs.

### Promoting a Result

```bash
./optimize.sh promote <path/to/best.json>
```

The `promote` command:
1. Validates the `best.json` artifact.
2. Creates a backup of the current registry (`config/pipeline.defaults.json.bak`).
3. Overwrites the registry with the optimized parameters.
4. Prints a diff of the changed values.

---

## Testing Optimized Configs (`--best`)

You can test an optimized configuration without promoting it globally using the
`--best` (or `--optimize-config`) flag on `dub.sh`:

```bash
./dub.sh input.mp4 en pt --best ~/.local/share/ai-dubbing/optimization/<run>/best.json
```

This injects the parameters as stage overrides for that specific run only. The
resulting workspace metadata will record the optimization source for
reproducibility.

---

## failure handling


Bad parameter combinations, CUDA OOM, model-load failures and interrupted
synthesis are caught per iteration, recorded as `FAILED` (with the error
string) in `history.jsonl`, and the loop continues. GPU memory is freed on
every path.

---

## Tunable parameters

The search space is defined in `src/optimization/parameter_space.py`:

| Parameter | Stage | Range | Effect |
|---|---|---|---|
| `samples.target_seconds` | samples | 6–14 s | Target speaker-reference length (identity). |
| `samples.max_seconds` | samples | 10–20 s | Max reference length. |
| `generate.duration_tolerance` | generate | 0.05–0.25 | Slack before the synthesizer fits by speed. |
| `generate.max_speed` | generate | 1.10–1.60 | Ceiling on speed-up used to fit a slot. |
| `generate.max_fit_iters` | generate | 1–4 | Synth/measure passes spent fitting duration. |
| `generate.use_clone_prompt` | generate | bool | Condition the clone on the reference transcript. |
| `align.tolerance` | align | 0.05–0.25 | When alignment resorts to time-stretching. |
| `align.min_abs_correction_s` | align | 0.05–0.30 | Absolute floor below which stretching is skipped. |

Add or adjust parameters by editing `_DEFAULT_PARAMETERS` in that module.

---

## Example: reference run

A 5-iteration English→English run on
`input/elon-musk-might-be-a-super-villain.mp4` (RTX 3060 Ti, ~100 s/iteration):

```
score progression: 0.884 → 0.905 → 0.908 → 0.946 → 0.938
best: 0.946 (iter 3)
```

The biggest gains came from lifting the weakest component — prosodic similarity
rose from `0.22` at baseline toward `0.55` — while speaker identity (`0.99`) and
reconstruction (`0.99`) stayed pinned. Best configuration:

```json
{
  "samples.target_seconds": 12.0,
  "samples.max_seconds": 12.0,
  "generate.duration_tolerance": 0.125,
  "generate.max_speed": 1.15,
  "generate.max_fit_iters": 4,
  "generate.use_clone_prompt": true,
  "align.tolerance": 0.075,
  "align.min_abs_correction_s": 0.125
}
```

---

## See also

* `docs/workspaces.md` — the workspace architecture the optimizer reuses.
* `docs/timing-aware-dubbing.md` — the timing model behind several metrics.
* `scripts/optimize_pipeline.py` — the Python entry point `optimize.sh` wraps
  (usable directly; `--help` for its flags).
