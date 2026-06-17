"""Tests for the search strategy, evaluator failure-handling, and the loop.

The optimizer is exercised with a GPU-free synthetic objective so convergence,
persistence and resume are validated without running the real pipeline.
"""
import random

from src.optimization.evaluator import EvaluationResult, PipelineEvaluator
from src.optimization.metrics import MetricResult
from src.optimization.optimizer import Optimizer, OptimizerConfig
from src.optimization.parameter_space import DEFAULT_SPACE
from src.optimization.search import HillClimbSearcher, rng_from_state, rng_to_state


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_first_proposal_is_baseline():
    s = HillClimbSearcher(DEFAULT_SPACE)
    p = s.propose(random.Random(0), best_config=None, stale_steps=0, have_baseline=False)
    assert p.source == "baseline"
    assert p.config == DEFAULT_SPACE.coerce_config(DEFAULT_SPACE.default_config())


def test_random_restart_after_patience():
    s = HillClimbSearcher(DEFAULT_SPACE, epsilon=0.0, restart_patience=3)
    best = DEFAULT_SPACE.default_config()
    p = s.propose(random.Random(0), best_config=best, stale_steps=5, have_baseline=True)
    assert p.source == "random"


def test_greedy_neighbor_when_not_stale():
    s = HillClimbSearcher(DEFAULT_SPACE, epsilon=0.0, restart_patience=99)
    best = DEFAULT_SPACE.default_config()
    p = s.propose(random.Random(0), best_config=best, stale_steps=0, have_baseline=True)
    assert p.source == "search"


def test_rng_state_roundtrip():
    rng = random.Random(42)
    [rng.random() for _ in range(5)]
    state = rng_to_state(rng)
    clone = rng_from_state(state)
    assert rng.random() == clone.random()


# ---------------------------------------------------------------------------
# Evaluator (injected runner/scorer — no GPU)
# ---------------------------------------------------------------------------


def _evaluator(tmp_path, runner=None, scorer=None):
    return PipelineEvaluator(
        input_path="/dev/null",
        source_language="en",
        target_language="en",
        baseline_config=DEFAULT_SPACE.coerce_config(DEFAULT_SPACE.default_config()),
        workspace_root=tmp_path,
        runner=runner or (lambda c, fs, ts: tmp_path),
        scorer=scorer,
    )


def test_evaluator_scores_ok(tmp_path):
    ev = _evaluator(tmp_path, scorer=lambda root: MetricResult(metrics={"x": 1.0}, composite=0.8))
    res = ev.evaluate(DEFAULT_SPACE.default_config())
    assert res.status == "ok" and res.score == 0.8


def test_evaluator_records_failure_and_continues(tmp_path):
    def boom(config, from_stage, to_stage):
        raise RuntimeError("CUDA out of memory")

    ev = _evaluator(tmp_path, runner=boom)
    res = ev.evaluate(DEFAULT_SPACE.default_config())
    assert res.status == "failed"
    assert "CUDA out of memory" in res.error


def test_from_stage_floor_is_generate(tmp_path):
    ev = _evaluator(tmp_path)
    # No change vs baseline → still synthesize from generate.
    assert ev._stages_to_run(DEFAULT_SPACE.default_config()) == "generate"
    # An align-only change starts at align.
    cfg = dict(DEFAULT_SPACE.default_config())
    cfg["align.tolerance"] = 0.2
    assert ev._stages_to_run(cfg) == "align"
    # A samples change starts at samples.
    cfg2 = dict(DEFAULT_SPACE.default_config())
    cfg2["samples.target_seconds"] = 8.0
    assert ev._stages_to_run(cfg2) == "samples"


def test_generate_cache_busted_when_generate_runs(tmp_path):
    gen = tmp_path / "generated_segments"
    gen.mkdir()
    man = gen / "manifest.json"
    man.write_text("[]")
    ev = _evaluator(tmp_path)
    ev.workspace_root = tmp_path
    ev.evaluate(DEFAULT_SPACE.default_config())  # from_stage=generate → bust
    assert not man.exists()


# ---------------------------------------------------------------------------
# Optimizer loop (synthetic objective)
# ---------------------------------------------------------------------------


def _synthetic_optimizer(tmp_path, iterations, seed=1):
    space = DEFAULT_SPACE
    optimum = space.default_config()  # baseline is the optimum here

    cfg = OptimizerConfig(
        input_path="/dev/null", language="en", iterations=iterations,
        run_dir=tmp_path / "run", seed=seed, epsilon=0.3, restart_patience=5,
    )
    ev = PipelineEvaluator(
        input_path="/dev/null", source_language="en", target_language="en",
        baseline_config=space.coerce_config(optimum),
        workspace_root=tmp_path, runner=lambda c, fs, ts: tmp_path,
    )

    def synthetic(config):
        dist = 0.0
        for k, v in config.items():
            p = space.get(k)
            if p.kind == "bool":
                dist += 0.0 if bool(v) == bool(optimum[k]) else 1.0
            else:
                dist += abs(float(v) - float(optimum[k])) / max(1e-9, p.high - p.low)
        score = max(0.0, 1.0 - dist / len(config))
        return EvaluationResult(status="ok", score=round(score, 6), metrics={"synthetic": score}, from_stage="generate")

    ev.evaluate = synthetic  # type: ignore[assignment]
    return Optimizer(cfg, evaluator=ev), cfg


def test_optimizer_runs_persists_and_finds_best(tmp_path):
    opt, cfg = _synthetic_optimizer(tmp_path, iterations=15)
    summary = opt.run()
    assert summary["n_iterations"] == 15
    # Baseline IS the optimum (score 1.0); the loop must find it on iter 0.
    assert summary["best_score"] == 1.0
    # History persisted to disk.
    assert (cfg.resolved_run_dir() / "history.jsonl").exists()
    assert (cfg.resolved_run_dir() / "best.json").exists()


def test_infinite_iterations_run_until_early_stop(tmp_path):
    # iterations<=0 means "infinite"; early-stop bounds it for the test.
    opt, cfg = _synthetic_optimizer(tmp_path, iterations=0)
    opt.config.early_stop_patience = 3
    summary = opt.run()
    # Baseline is the optimum (1.0); 3 non-improving steps then stop → 4 total.
    assert summary["best_score"] == 1.0
    assert summary["n_iterations"] >= 4
    assert "workspace_root" in summary


def test_optimizer_resumes_without_repeating_iterations(tmp_path):
    opt1, cfg = _synthetic_optimizer(tmp_path, iterations=5)
    opt1.run()
    # Re-run with more iterations: should continue from 5, not restart.
    opt2, cfg2 = _synthetic_optimizer(tmp_path, iterations=9)
    summary = opt2.run()
    iters = [r.iteration for r in opt2.history.records()]
    assert iters == list(range(9))  # contiguous, no duplicates
    assert summary["n_iterations"] == 9
