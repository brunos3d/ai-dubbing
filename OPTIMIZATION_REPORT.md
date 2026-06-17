# Pipeline Self-Optimization — Full Report

> An automatic experiment runner that continuously improves the dubbing
> pipeline by re-processing the same video and measuring how closely the output
> matches the source. Not a model trainer — a parameter optimizer.

---

## 1. Executive summary

The dubbing pipeline has many tunable knobs (speaker-reference length, synthesis
speed/fitting, alignment thresholds, …) whose best values were previously chosen
by hand. This subsystem turns that into a **measurable, automatic search**:

1. Run the **same** clip through the pipeline in a **same-language benchmark**
   (e.g. English → English), where the words are known so the output can be
   compared directly against the source.
2. Score the result with a **composite of 8 objective quality metrics**.
3. **Search** the parameter space (hill-climbing + random exploration) for the
   configuration that maximises that score.
4. **Persist** every result; the run is autonomous, failure-tolerant, resumable,
   and bounded in disk use.

On the Elon/Colbert reference clip (EN→EN, OmniVoice, RTX 3060 Ti), a short
5-iteration run lifted the composite score from **0.884 → 0.946**, driven almost
entirely by fixing the pipeline's measurably weakest behaviour — prosodic
similarity (`0.22 → 0.74`). The system surfaced *which* subsystem was weak,
*quantified* it, and *found a configuration that fixes it* — with no human
intervention.

---

## 2. What it does

### 2.1 The same-language benchmark

Optimization needs a ground truth. In normal dubbing (EN→PT) there is no
reference for "what the Portuguese should sound like". So we optimize on the
**controllable** case where source and target language are identical:

| Property | Why same-language makes it measurable |
|---|---|
| Semantic content | Known — the dub *is* the source text, so fidelity is comparable. |
| Timing | The source slot durations are the exact target. |
| Speaker identity | The original speaker audio is the reference to clone against. |
| Acoustics | Same words ⇒ spectral envelope / energy contour should track the source. |

A pipeline tuned to reproduce the source faithfully in the same-language case
carries its improved timing/identity/prosody behaviour over to real
multilingual dubbing.

### 2.2 The loop

```
        ┌──────────────────────────────────────────────────────────┐
        │  prepare()  — analysis phase, run ONCE                     │
        │  extract → separate → diarize → samples → transcribe →     │
        │  translate   (cached; independent of the tuned parameters) │
        └──────────────────────────────────────────────────────────┘
                                   │
            ┌──────────────────────┴───────────────────────┐
            │             per-iteration loop                │
            │                                               │
   search.propose() ──►  config  ──►  evaluator.evaluate()  │
            ▲                              │                 │
            │                   re-run only affected stages: │
            │              (generate → align → reconstruct)  │
            │                              │                 │
            │                       compute_metrics()        │
            │                              │                 │
        best/stale ◄── history.append() ◄─ score            │
            └───────────────────────────────────────────────┘
```

Only the stages **downstream of a changed parameter** re-run each iteration; the
expensive analysis phase (Demucs, pyannote, Whisper, translation) runs once and
is reused. This is what makes hundreds of iterations practical.

---

## 3. Architecture & files

### Files created

| File | Lines | Role |
|---|---:|---|
| `src/optimization/__init__.py` | 53 | Package surface / docs. |
| `src/optimization/parameter_space.py` | 228 | Tunable parameters, ranges, sampling, perturbation, param→stage mapping. |
| `src/optimization/metrics.py` | 499 | The 8 quality metrics + composite scoring. |
| `src/optimization/evaluator.py` | 220 | Runs the pipeline for one config; failure-safe; cache-busting. |
| `src/optimization/search.py` | 98 | Hill-climbing + epsilon-random proposer; RNG (de)serialisation. |
| `src/optimization/history.py` | 152 | Append-only JSONL history, best tracking, resume. |
| `src/optimization/optimizer.py` | 229 | The autonomous, resumable loop. |
| `scripts/optimize_pipeline.py` | 192 | Python CLI entry point. |
| `optimize.sh` | 271 | Terminal front-end (live status, subcommands). |
| `docs/optimize.md` | 231 | Usage guide for `optimize.sh`. |
| `tests/test_optimization_parameter_space.py` | — | Parameter-space tests. |
| `tests/test_optimization_metrics.py` | — | Metric tests on a synthetic mini-workspace. |
| `tests/test_optimization_history.py` | — | History persistence / resume tests. |
| `tests/test_optimization_search_optimizer.py` | — | Search, evaluator (mock), and loop tests. |

### Files modified

| File | Change |
|---|---|
| `src/workspace/pipeline.py` | Added a `stage_overrides` seam: `WorkspacePipeline(stage_overrides=…)` → applied in `_build_stage` → surfaced in `_current_configs` so the invalidation DAG re-runs affected stages. Excluded from the workspace-identity hash so tuning reuses one workspace. |
| `tests/test_workspace_e2e.py`, `tests/test_workspace_edit_scenarios.py` | Stubs updated to accept/apply the new `stage_overrides` kwarg. |
| `README.md` | Added a "Self-optimization" section linking `docs/optimize.md`. |

**Tests:** 30 new optimization tests; **242/242** pass overall.

---

## 4. Optimization strategy

**Selected: hill-climbing with epsilon-random exploration and random restarts.**

| Considered | Verdict |
|---|---|
| Brute force / grid | Rejected — evaluation is expensive (real synthesis, ~100 s each). |
| **Hill-climbing + ε-random** | **Chosen** — improves from the current best with few evaluations; trivially resumable; no extra deps. |
| Simulated annealing / evolutionary | Overkill for a small, mostly-separable space at this stage. |
| Bayesian optimization | Best long-term (sample-efficient) but needs a surrogate-model dependency — noted as future work. |

Mechanics (`search.py`):
- The **first** proposal is always the **baseline** (default config) — a fixed
  reference point every run is anchored to.
- Otherwise: a **greedy neighbour** of the current best (perturb 1 parameter by
  one grid step), except with probability `epsilon` (default 0.2) or after
  `restart_patience` (default 8) non-improving steps, take a **random** config
  to escape local optima.
- The proposer is **pure** given `(seed, history)`; the RNG state is persisted,
  so a stopped run resumes deterministically.

---

## 5. Metrics implemented

Composite score ∈ `[0, 1]` (higher = better) = weighted mean of the components
below; a metric that can't be computed is dropped and the weights renormalise.

| Metric | Weight | What it measures | What a low value means |
|---|---:|---|---|
| `timing_accuracy` | 0.18 | Final segment duration vs source slot. | Segments too long/short. |
| `slot_fit` | 0.12 | Overrun penalty (speech past its slot). | Turns bleed into the next. |
| `speaker_similarity` | 0.18 | Cloned timbre vs real speaker reference (MFCC cosine). | Voice doesn't match the speaker. |
| `speaker_consistency` | 0.10 | Intra-speaker cohesion across segments. | Identity drifts / attribution swaps. |
| `ending_quality` | 0.10 | Trailing-energy ratio per segment. | Clipped / abruptly-cut endings. |
| `continuity` | 0.12 | Temporal collisions between consecutive turns. | Overlapping, muddled transitions. |
| `prosodic_similarity` | 0.10 | Source vs generated energy contour (Pearson). | Emphasis/excitement flattened. |
| `reconstruction_quality` | 0.10 | Global duration + log-mel envelope vs source speech. | Overall dub diverges from source. |

All metrics are numpy + (optional) librosa — CPU-only, no torch, no network —
and read only artifacts the pipeline already produces. Weights live in
`metrics.py::DEFAULT_WEIGHTS`.

---

## 6. Parameters optimized

Discovered by inspecting the stages directly (`parameter_space.py`):

| Parameter | Stage | Range | Quality lever |
|---|---|---|---|
| `samples.target_seconds` | samples | 6–14 s | Speaker-reference length (identity). |
| `samples.max_seconds` | samples | 10–20 s | Reference length ceiling. |
| `generate.duration_tolerance` | generate | 0.05–0.25 | Slack before fitting by speed. |
| `generate.max_speed` | generate | 1.10–1.60 | Speed-up ceiling for slot fitting. |
| `generate.max_fit_iters` | generate | 1–4 | Synth/measure fitting passes. |
| `generate.use_clone_prompt` | generate | bool | Condition clone on reference transcript. |
| `align.tolerance` | align | 0.05–0.25 | When alignment time-stretches. |
| `align.min_abs_correction_s` | align | 0.05–0.30 | Absolute floor below which stretching is skipped. |

**Two correctness mechanisms make this safe and efficient:**

- **Incremental re-runs.** A changed parameter maps to its stage; the evaluator
  passes `from_stage` = the earliest affected stage, so only that stage and its
  downstream re-run (e.g. a `generate` change re-runs generate→align→reconstruct;
  an `align` change re-runs only align→reconstruct). Upstream analysis is reused.
- **Cache busting.** The generate stage reuses segments by a render-key that does
  *not* encode the synthesis-fitting parameters. The evaluator deletes the
  generated-segments manifest whenever generate is scheduled, forcing fresh
  synthesis so those parameters actually take effect.

---

## 7. Storage & safety

- **Bounded disk.** Every iteration reuses **one** workspace (tuned parameters
  are excluded from the workspace-identity hash) and overwrites segment audio in
  place — footprint is roughly constant, not growing per iteration. The
  optimization run directory stores only small JSON.
- **Resumable.** `history.jsonl` (one record/iteration), `best.json`, and
  `state.json` (RNG + counters + workspace root) are written every iteration.
  Ctrl-C and re-run the same command to continue exactly where it left off.
- **Failure-tolerant.** Bad parameter combos, CUDA OOM, model-load failures, and
  interrupted synthesis are caught, recorded as `FAILED` with the error string,
  and the loop continues. GPU memory is freed on every path.

Run artifacts live under `~/.local/share/ai-dubbing/optimization/<run-name>/`
(parent overridable via `AI_DUBBING_OPT_ROOT`).

---

## 8. Results — reference run

EN→EN on `input/elon-musk-might-be-a-super-villain.mp4`, OmniVoice, RTX 3060 Ti,
~100 s/iteration, 5 iterations, 0 failures.

**Score progression:** `0.884 → 0.905 → 0.908 → 0.946 → 0.938`
**Best:** `0.946` at iteration 3.

**Baseline vs best — per metric:**

| Metric | Baseline | Best | Δ |
|---|---:|---:|---:|
| timing_accuracy | 0.866 | 0.884 | +0.018 |
| slot_fit | 0.954 | 0.973 | +0.019 |
| speaker_similarity | 0.987 | 0.989 | +0.002 |
| speaker_consistency | 0.989 | 0.992 | +0.003 |
| ending_quality | 1.000 | 1.000 | +0.000 |
| continuity | 0.967 | 1.000 | +0.033 |
| **prosodic_similarity** | **0.222** | **0.737** | **+0.516** |
| reconstruction_quality | 0.994 | 0.995 | +0.001 |

**Best configuration discovered:**

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

**Reading the result:** the system identified prosodic similarity as the
dominant weakness (baseline `0.22`) and found that a *lower* `max_speed` (1.15
vs 1.35) with *more* fitting passes (4 vs 2) preserves the source energy contour
far better — without sacrificing speaker identity (0.99) or reconstruction
(0.99). That is a concrete, data-backed pipeline insight produced automatically.

---

## 9. How to use this to improve the project

This is the payoff: the optimizer is a **measurement and decision instrument**,
not just a tuner. Concrete workflows:

### 9.1 Diagnose the weakest subsystem objectively
Run `./optimize.sh status …` (or read `best.json`) and look at the per-metric
breakdown. The lowest component tells you, with a number, *where the pipeline is
worst* — no subjective listening required. In the reference run that was
prosody; the metric pointed straight at it.

### 9.2 Promote discovered configurations to defaults
A winning config is a recipe. Fold the best values into the stage defaults
(`GenerateStage`, `AlignStage`, `SampleStage` constructors) so **every** user
benefits — then re-run the optimizer to confirm the new baseline scores at the
discovered level. The same-language score becomes the acceptance criterion.

### 9.3 Quality gate / regression guardrail
Wire a short bounded run (or a single baseline evaluation) into CI:
`compute_metrics()` on a fixed clip yields a composite score; fail the build if
it drops more than a threshold below the recorded best. This catches quality
regressions that unit tests can't (a refactor that quietly clips endings or
breaks prosody shows up as a metric drop).

### 9.4 Metric-driven development
Each metric maps to a code area, turning "improve quality" into targeted work:

| If this metric is low… | …investigate |
|---|---|
| `timing_accuracy` / `slot_fit` | `timing/duration.py`, `align.py` thresholds, `fit_by_speed`. |
| `speaker_similarity` / `_consistency` | `samples.py` reference selection/purification, diarization stability. |
| `ending_quality` | synthesis tail handling, segment trimming. |
| `continuity` | segment placement / overlap in `reconstruct.py`. |
| `prosodic_similarity` | `timing/prosody.py`, `generate` speed mapping. |
| `reconstruction_quality` | end-to-end placement, sample-rate handling. |

You can also re-weight `DEFAULT_WEIGHTS` to make the search *prioritise* whatever
the project currently cares about most.

### 9.5 A/B-test pipeline changes
Made a code change to a stage? Run the optimizer (or a baseline eval) before and
after on the same clip. The composite score (and per-metric deltas) tell you
whether the change actually helped — and by how much — instead of guessing.

### 9.6 Expose more knobs
The space is intentionally small but extensible. Add a parameter to
`_DEFAULT_PARAMETERS` (e.g. denoise strength, diarization margin, prosody bands,
OmniVoice prompt formatting) and the search picks it up automatically — provided
the stage reads it as a simple attribute. This is how the search space grows as
the team identifies new candidate levers.

### 9.7 Per-dataset / per-content profiles
Different content (interview vs. monologue vs. multi-speaker panel) may favour
different parameters. Run separate named runs per representative clip
(`-r interview-en`, `-r monologue-en`) and keep a small library of best configs
selected by content type.

### 9.8 Bridge to multilingual
Once same-language scores are high and stable, validate transfer: dub EN→PT with
the discovered config and spot-check. The hypothesis (and the reason for the
whole same-language approach) is that timing/identity/prosody gains are
language-agnostic. Discrepancies become the next research question.

---

## 10. Limitations

- **Acoustic metrics are proxies, not perceptual ground truth.** MFCC-cosine
  speaker similarity and log-mel envelope correlation are robust and CPU-cheap
  but coarser than a learned speaker-verification embedding or a MOS predictor.
- **Prosody metric is energy-only.** It correlates source vs generated energy
  contour; pitch-contour and rhythm correlation are not yet folded in.
- **Same-language translation is near-identity** via `deep-translator`; it
  exercises timing/synthesis/reconstruction well but does not stress the
  translation-candidate selector (which matters for true cross-lingual fitting).
- **Greedy search** can plateau in local optima; ε-random + restarts mitigate
  but don't guarantee global optimality, and there is no surrogate model yet.
- **Cost.** Each generate-touching iteration is a full re-synthesis
  (~100 s here); large sweeps are hours-scale (by design — it's built to run
  unattended for a long time).
- **Speaker-identity metrics reuse the same reference** the clone was built
  from, so similarity is somewhat self-referential; an independent held-out
  reference clip would be stronger.

---

## 11. Next recommended improvements

1. **Stronger metrics:** swap MFCC similarity for a pretrained speaker-embedding
   (e.g. the pyannote/WeSpeaker model already in the diarizer) and add a learned
   MOS/naturalness predictor; add pitch-contour correlation to prosody.
2. **Bayesian optimization** (e.g. a Gaussian-process or TPE surrogate) for
   far better sample efficiency on the expensive evaluations.
3. **Multi-clip objective:** average the score over several representative clips
   per run so configs generalise rather than overfit one video.
4. **Auto-promotion + CI gate:** a command that writes the best config back into
   stage defaults behind a flag, plus a CI check on the composite score.
5. **Expand the search space:** denoise strength, diarization `DEFAULT_K_MARGIN`,
   prosody speed/gain bands, reference gap/edge-pad, candidate-selection weights.
6. **Parallel / distributed evaluation** across GPUs to shorten wall-clock for
   large sweeps.
7. **Held-out speaker reference** for an unbiased identity metric.
8. **Held-out multilingual validation** to quantify same-language → cross-lingual
   transfer empirically.

---

## 12. Appendix — commands

```bash
# Run (infinite, resumable)
./optimize.sh run    -m input/elon-musk-might-be-a-super-villain.mp4 -l en
# Bounded run
./optimize.sh run    -m input/clip.mp4 -l en -n 50
# Inspect
./optimize.sh status -m input/clip.mp4 -l en
./optimize.sh best   -m input/clip.mp4 -l en
./optimize.sh watch  -m input/clip.mp4 -l en      # live pipeline log
# GPU-free smoke test of the loop
./optimize.sh run    -m input/clip.mp4 -l en --mock -n 30
```

See `docs/optimize.md` for the full flag reference.
```bash
# Run the test suite (30 optimization tests; 242 total)
.venv/bin/python -m pytest tests/test_optimization_*.py -q
```
