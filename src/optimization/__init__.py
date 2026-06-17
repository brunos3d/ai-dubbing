"""Automatic pipeline-optimization subsystem.

This is **not** a machine-learning trainer. It is an automated experiment
runner that repeatedly processes the *same* video through the existing dubbing
pipeline, measures how closely the generated output matches the source
material, and searches for pipeline parameter configurations that maximise that
measured quality.

The intended benchmark is **same-language dubbing** (e.g. English → English):
when source and target language are identical the semantic content is known, so
timing, speaker identity, prosody and reconstruction can all be compared
directly against the source. A pipeline tuned on the controllable same-language
case generalises to the multilingual case.

Layout
------
* :mod:`~src.optimization.parameter_space` — the tunable parameters, their
  ranges, sampling and perturbation, and the param→stage mapping that tells the
  evaluator which pipeline stage to re-run.
* :mod:`~src.optimization.metrics` — the measurable quality components and the
  composite score.
* :mod:`~src.optimization.evaluator` — runs the pipeline for one configuration
  and turns its artifacts into a score (failure-safe).
* :mod:`~src.optimization.history` — append-only, resumable run history.
* :mod:`~src.optimization.search` — the search strategies (hill-climbing with
  epsilon-random exploration).
* :mod:`~src.optimization.optimizer` — ties it all together into an autonomous,
  resumable loop.
"""
from __future__ import annotations

from .history import HistoryStore, IterationRecord
from .metrics import MetricResult, compute_metrics, score_from_metrics
from .parameter_space import (
    DEFAULT_SPACE,
    ParameterSpace,
    Parameter,
    earliest_stage,
    to_stage_overrides,
)

__all__ = [
    "DEFAULT_SPACE",
    "ParameterSpace",
    "Parameter",
    "earliest_stage",
    "to_stage_overrides",
    "MetricResult",
    "compute_metrics",
    "score_from_metrics",
    "HistoryStore",
    "IterationRecord",
]
