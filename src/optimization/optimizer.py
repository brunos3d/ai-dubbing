"""The autonomous, resumable optimization loop.

Ties together the parameter space, evaluator, search strategy and history into
the experiment runner described in the brief:

    select config → run pipeline → evaluate → score → persist → adjust → repeat

with no human intervention, surviving failures, and bounded in disk use. A run
can be stopped and re-run later; it continues from the persisted state and best
configuration.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from ..utils.logging import get_logger
from ..utils.vram import free_vram_aggressive, log_vram
from ..workspace.paths import slugify, workspaces_root
from .evaluator import PipelineEvaluator
from .history import HistoryStore, IterationRecord
from .metrics import DEFAULT_WEIGHTS
from .parameter_space import DEFAULT_SPACE, ParameterSpace
from .search import HillClimbSearcher, rng_from_state, rng_to_state

LOG = get_logger("ai-dubbing.optimize")


def _now_iso() -> str:
    import datetime as _dt

    return _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def optimization_root() -> Path:
    """Directory that holds all optimization runs (sibling of ``workspaces/``).

    Honours ``AI_DUBBING_OPT_ROOT`` for tests / overrides.
    """
    import os

    override = os.environ.get("AI_DUBBING_OPT_ROOT")
    if override:
        return Path(override)
    return workspaces_root().parent / "optimization"


@dataclass
class OptimizerConfig:
    """Everything a run needs. Same-language by default (``source==target``)."""

    input_path: str
    language: str = "en"
    iterations: int = 25
    run_name: Optional[str] = None
    run_dir: Optional[Path] = None
    seed: int = 1234
    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    to_stage: str = "reconstruct"
    hf_token: Optional[str] = None
    whisper_model: str = "large-v3"
    no_pyannote: bool = False
    epsilon: float = 0.2
    neighbors_k: int = 1
    restart_patience: int = 8
    early_stop_patience: Optional[int] = None  # stop after N non-improving iters
    workspace_root: Optional[Path] = None  # mainly for tests

    def resolved_run_dir(self) -> Path:
        if self.run_dir is not None:
            return Path(self.run_dir)
        name = self.run_name or (
            f"{slugify(Path(self.input_path).stem)}-{self.language}-{self.language}"
        )
        return optimization_root() / name


class Optimizer:
    """Drive the search loop over a single reused same-language workspace."""

    def __init__(
        self,
        config: OptimizerConfig,
        *,
        space: ParameterSpace = DEFAULT_SPACE,
        evaluator: Optional[PipelineEvaluator] = None,
    ) -> None:
        self.config = config
        self.space = space
        self.history = HistoryStore(config.resolved_run_dir())
        self.searcher = HillClimbSearcher(
            space,
            epsilon=config.epsilon,
            neighbors_k=config.neighbors_k,
            restart_patience=config.restart_patience,
        )
        self.evaluator = evaluator or PipelineEvaluator(
            input_path=config.input_path,
            source_language=config.language,
            target_language=config.language,
            baseline_config=space.coerce_config(space.default_config()),
            weights=config.weights,
            to_stage=config.to_stage,
            hf_token=config.hf_token,
            whisper_model=config.whisper_model,
            no_pyannote=config.no_pyannote,
            workspace_root=config.workspace_root,
        )

    # -- resume helpers ---------------------------------------------------

    def _load_rng(self) -> random.Random:
        state = self.history.load_state().get("rng")
        if state is not None:
            try:
                return rng_from_state(state)
            except Exception:  # noqa: BLE001 - corrupt state → fresh seed
                pass
        return random.Random(self.config.seed)

    # -- main loop --------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        cfg = self.config
        infinite = not cfg.iterations or cfg.iterations <= 0
        LOG.info(
            "Optimization run dir: %s | target iterations: %s | %s→%s",
            self.history.run_dir,
            ("infinite" if infinite else cfg.iterations),
            cfg.language, cfg.language,
        )

        # One-time analysis phase (idempotent; cheap on resume).
        LOG.info("Preparing workspace (analysis phase)…")
        root = self.evaluator.prepare()
        LOG.info("Workspace ready: %s", root)

        resume = self.history.resume()
        rng = self._load_rng()
        state = resume["state"]
        best = resume["best"]
        best_config: Optional[Dict[str, Any]] = best.parameters if best else None
        best_score: Optional[float] = best.score if best else None
        stale_steps = int(state.get("stale_steps", 0))
        have_baseline = any(
            r.source == "baseline" or r.status == "ok" for r in resume["records"]
        )
        start_iter = resume["next_iteration"]
        if start_iter > 0:
            LOG.info(
                "Resuming from iteration %d (best so far: %s)",
                start_iter, (round(best_score, 5) if best_score is not None else None),
            )

        no_improve = 0
        i = start_iter
        try:
            while infinite or i < cfg.iterations:
                proposal = self.searcher.propose(
                    rng,
                    best_config=best_config,
                    stale_steps=stale_steps,
                    have_baseline=have_baseline,
                )
                config = self.space.coerce_config(proposal.config)
                LOG.info("[iter %d] %s config: %s", i, proposal.source, config)

                result = self.evaluator.evaluate(config)
                
                # Log VRAM usage after each iteration to track memory leaks
                log_vram(LOG)
                
                # Aggressively free VRAM between iterations to prevent accumulation
                free_vram_aggressive()
                
                record = IterationRecord(
                    iteration=i,
                    score=result.score,
                    parameters=config,
                    metrics=result.metrics,
                    status=result.status,
                    error=result.error,
                    from_stage=result.from_stage,
                    duration_s=result.duration_s,
                    timestamp=_now_iso(),
                    source=proposal.source,
                )
                self.history.append(record)
                have_baseline = have_baseline or proposal.source == "baseline"

                improved = (
                    result.status == "ok"
                    and result.score is not None
                    and (best_score is None or result.score > best_score)
                )
                if improved:
                    best_score = result.score
                    best_config = config
                    stale_steps = 0
                    no_improve = 0
                    LOG.info("[iter %d] NEW BEST score=%.5f", i, best_score)
                else:
                    stale_steps += 1
                    no_improve += 1
                    status = result.status if result.status == "failed" else "no improvement"
                    LOG.info(
                        "[iter %d] %s (score=%s, best=%s)",
                        i, status,
                        (round(result.score, 5) if result.score is not None else None),
                        (round(best_score, 5) if best_score is not None else None),
                    )

                self.history.save_state({
                    "rng": rng_to_state(rng),
                    "stale_steps": stale_steps,
                    "last_iteration": i,
                    "best_score": best_score,
                    "workspace_root": str(self.evaluator.workspace_root or ""),
                    "language": cfg.language,
                    "updated_at": _now_iso(),
                })

                if cfg.early_stop_patience and no_improve >= cfg.early_stop_patience:
                    LOG.info(
                        "Early stop: %d iterations without improvement", no_improve
                    )
                    break
                i += 1
        except KeyboardInterrupt:
            LOG.info("Interrupted by user. Run is resumable.")
            # Save state before exiting
            self.history.save_state({
                "rng": rng_to_state(rng),
                "stale_steps": stale_steps,
                "last_iteration": i - 1 if i > start_iter else start_iter,
                "best_score": best_score,
                "workspace_root": str(self.evaluator.workspace_root or ""),
                "language": cfg.language,
                "updated_at": _now_iso(),
            })
            raise

        summary = self.history.summary()
        summary["workspace_root"] = str(self.evaluator.workspace_root or "")
        LOG.info(
            "Run complete. best_score=%s at iteration %s",
            summary.get("best_score"), summary.get("best_iteration"),
        )
        return summary
