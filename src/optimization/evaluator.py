"""Run the pipeline for one parameter configuration and score the result.

The evaluator owns the single, reused optimization workspace. It:

1. runs the analysis phase **once** (:meth:`prepare`) — the expensive
   ``extract … translate`` stages that do not depend on the tuned parameters;
2. for each configuration, re-runs only the **affected** stages
   (``from_stage`` = the earliest stage any changed parameter touches), then
   scores the workspace with :func:`src.optimization.metrics.compute_metrics`.

Two design points worth calling out:

* **Bounded disk.** The same workspace is reused for every iteration (the tuned
  parameters are excluded from the workspace-identity hash), and segment audio
  is overwritten in place, so the workspace footprint is constant rather than
  growing per iteration.

* **Cache busting.** The generate stage reuses previously-synthesized segments
  keyed by a render-key that does *not* encode the synthesis-fitting parameters
  (``max_speed`` / ``duration_tolerance`` / ``max_fit_iters`` /
  ``use_clone_prompt``). To make those parameters actually take effect we delete
  the generated-segments manifest whenever the generate stage is scheduled to
  run, forcing a fresh synthesis. Align-only iterations leave the generated
  segments untouched (fast).

Failure handling: any exception (bad parameter combo, CUDA OOM, model load
failure, interrupted synthesis) is caught, recorded as a ``failed`` result with
the error string, and the loop continues. GPU memory is freed on every path.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..utils.logging import get_logger
from ..workspace.dag import STAGE_ORDER
from .metrics import DEFAULT_WEIGHTS, MetricResult, compute_metrics
from .parameter_space import changed_params, earliest_stage, to_stage_overrides

LOG = get_logger("ai-dubbing.optimizer.evaluator")


def _is_oom(exc: Exception) -> bool:
    """Check if an exception is a CUDA out-of-memory error."""
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda" in msg and "oom" in msg


def _free_vram_quiet() -> None:
    """Free VRAM without logging."""
    try:
        from ..utils.vram import free_vram
        free_vram()
    except Exception:
        pass


def _free_vram_aggressive_quiet() -> None:
    """Aggressively free VRAM without logging."""
    try:
        from ..utils.vram import free_vram_aggressive
        free_vram_aggressive()
    except Exception:
        pass


def _log_vram_quiet() -> None:
    """Log current VRAM usage."""
    try:
        from ..utils.vram import log_vram
        log_vram(LOG)
    except Exception:
        pass


@dataclass
class EvaluationResult:
    """The outcome of evaluating one configuration."""

    status: str  # "ok" | "failed"
    score: Optional[float] = None
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    from_stage: Optional[str] = None
    duration_s: Optional[float] = None


# Type of an injected runner: (config, from_stage, to_stage) -> workspace_root.
RunnerFn = Callable[[Dict[str, Any], Optional[str], str], Path]
ScorerFn = Callable[[Path], MetricResult]


def _free_vram_quiet() -> None:
    try:
        from ..utils.vram import free_vram

        free_vram()
    except Exception:  # noqa: BLE001
        pass


class PipelineEvaluator:
    """Evaluate configurations against one reused same-language workspace."""

    def __init__(
        self,
        *,
        input_path: str,
        source_language: str,
        target_language: str,
        baseline_config: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
        to_stage: str = "reconstruct",
        hf_token: Optional[str] = None,
        whisper_model: str = "large-v3",
        no_pyannote: bool = False,
        workspace_root: Optional[Path] = None,
        runner: Optional[RunnerFn] = None,
        scorer: Optional[ScorerFn] = None,
    ) -> None:
        self.input_path = input_path
        self.source_language = source_language
        self.target_language = target_language
        self.baseline_config = dict(baseline_config)
        self.weights = weights or DEFAULT_WEIGHTS
        self.to_stage = to_stage
        self.hf_token = hf_token
        self.whisper_model = whisper_model
        self.no_pyannote = no_pyannote
        self._workspace_root_override = workspace_root
        # Injection seams for tests (avoid GPU/models entirely).
        self._runner = runner
        self._scorer = scorer or (lambda root: compute_metrics(root, weights=self.weights))
        self.workspace_id: Optional[str] = None
        self.workspace_root: Optional[Path] = None

    # -- pipeline construction -------------------------------------------

    def _make_pipeline(self, config: Dict[str, Any]):
        from ..workspace.pipeline import WorkspacePipeline

        return WorkspacePipeline(
            self.input_path,
            self.source_language,
            self.target_language,
            whisper_model=self.whisper_model,
            hf_token=self.hf_token,
            no_pyannote=self.no_pyannote,
            tts_backend="omnivoice",
            stage_overrides=to_stage_overrides(config),
            workspace_root=self._workspace_root_override,
        )

    # -- phase 1: one-time analysis --------------------------------------

    def prepare(self) -> Path:
        """Run the analysis phase once; return the workspace root.

        With an injected runner (tests) this is a no-op that just resolves the
        workspace root.
        """
        if self._runner is not None:
            root = self._workspace_root_override or Path(".")
            self.workspace_root = Path(root)
            return self.workspace_root
        wp = self._make_pipeline(self.baseline_config)
        wid, root = wp.prepare()
        self.workspace_id = wid
        self.workspace_root = root
        return root

    # -- phase 2: per-config evaluation ----------------------------------

    def _stages_to_run(self, config: Dict[str, Any]) -> Optional[str]:
        """``from_stage`` for this config: the earliest stage any changed
        parameter touches.

        * change before generate (e.g. ``samples``) → start there so synthesis
          re-runs against the new upstream artifacts;
        * change at generate → re-synthesize;
        * change after generate (e.g. ``align``) → start at that stage and reuse
          the existing generated segments (fast);
        * nothing changed (baseline) → default to ``generate`` because the
          analysis phase only ran through ``translate``.
        """
        changed = changed_params(config, self.baseline_config)
        return earliest_stage(changed) or "generate"

    def _bust_generate_cache(self) -> None:
        """Delete the generated-segments manifest so generate re-synthesizes.

        Required because the per-segment reuse key does not encode the
        synthesis-fitting parameters; without this, tuning them would be a
        no-op.
        """
        if self.workspace_root is None:
            return
        man = self.workspace_root / "generated_segments" / "manifest.json"
        try:
            if man.exists():
                man.unlink()
        except OSError:
            pass

    def _default_run(self, config: Dict[str, Any], from_stage: Optional[str], to_stage: str) -> Path:
        wp = self._make_pipeline(config)
        wid = self.workspace_id
        if wid is None:
            wid, _, _ = wp._id()
            self.workspace_id = wid

        gen_idx = STAGE_ORDER.index("generate")
        if from_stage is not None and STAGE_ORDER.index(from_stage) < gen_idx:
            # A parameter *before* generate changed (e.g. samples.*). Re-run only
            # the changed upstream analysis that generate consumes, stopping at
            # ``samples`` — transcribe (Whisper-large) and translate (network) do
            # NOT depend on those parameters, so re-running them would waste a
            # heavy model load per iteration and add VRAM pressure that triggers
            # CUDA OOM on long runs. Then synthesize from generate onward.
            wp.generate(wid, from_stage=from_stage, to_stage="samples")
            wp.generate(wid, from_stage="generate", to_stage=to_stage)
        else:
            wp.generate(wid, from_stage=from_stage, to_stage=to_stage)
        return self.workspace_root  # type: ignore[return-value]

    def evaluate(self, config: Dict[str, Any]) -> EvaluationResult:
        """Run the pipeline for ``config`` and score it. Never raises.

        A CUDA out-of-memory failure is retried **once** after aggressively
        freeing VRAM: on a long-lived run the 8 GB card fragments across
        repeated model load/free cycles, so a cleared retry usually succeeds.
        A second OOM (or any other error) is recorded as ``failed`` and the
        loop moves on.
        """
        # Aggressively free VRAM at the start of each iteration to prevent
        # accumulation across multiple evaluations
        _free_vram_aggressive_quiet()
        
        from_stage = self._stages_to_run(config)
        started = time.time()
        runner = self._runner or self._default_run
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                # If the generate stage is in the run window, force fresh synthesis.
                if from_stage is not None and STAGE_ORDER.index(from_stage) <= STAGE_ORDER.index("generate"):
                    self._bust_generate_cache()

                root = runner(config, from_stage, self.to_stage)
                self.workspace_root = Path(root)
                result = self._scorer(self.workspace_root)
                _free_vram_aggressive_quiet()
                return EvaluationResult(
                    status="ok",
                    score=result.composite,
                    metrics=result.metrics,
                    details=result.details,
                    from_stage=from_stage,
                    duration_s=round(time.time() - started, 2),
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - record and continue
                _free_vram_aggressive_quiet()
                if attempt + 1 < max_attempts and _is_oom(exc):
                    LOG.warning(
                        "CUDA OOM on attempt %d for config %s; freed VRAM, retrying once",
                        attempt + 1, from_stage,
                    )
                    continue
                err = f"{type(exc).__name__}: {exc}"
                tb = traceback.format_exc().strip().splitlines()
                detail = tb[-1] if tb else err
                return EvaluationResult(
                    status="failed",
                    score=None,
                    error=err,
                    details={"trace_tail": detail},
                    from_stage=from_stage,
                    duration_s=round(time.time() - started, 2),
                )
        # Unreachable (the loop either returns or raises), but keeps mypy happy.
        return EvaluationResult(status="failed", score=None, from_stage=from_stage)
