"""The tunable parameter space and its sampling / perturbation operators.

Every parameter is named ``"<stage>.<attr>"`` where ``<stage>`` is a pipeline
stage (``samples`` / ``generate`` / ``align`` / ``reconstruct`` …) and
``<attr>`` is the stage attribute the optimizer overrides via
``WorkspacePipeline(stage_overrides=...)``.  The names line up with each
stage's declared ``config_fields`` so a change reliably invalidates the stage
through the workspace DAG.

The parameters here were discovered by inspecting the stages directly:

* ``samples.target_seconds`` / ``samples.max_seconds`` — speaker-reference
  window length (identity preservation, Objective #1).
* ``generate.duration_tolerance`` — slack before the synthesizer fits by speed.
* ``generate.max_speed`` — ceiling on speed-up used to fit a slot
  (timing vs naturalness trade-off, Objective #3).
* ``generate.max_fit_iters`` — how many synth/measure passes to spend fitting.
* ``generate.use_clone_prompt`` — whether the reference transcript conditions
  the clone.
* ``align.tolerance`` / ``align.min_abs_correction_s`` — when the align stage
  resorts to (lossy) time-stretching to hit the slot.

Only OmniVoice is used as the TTS backend for the self-improvement system, so
no F5-TTS-specific parameters are exposed.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from ..workspace.dag import STAGE_ORDER


@dataclass(frozen=True)
class Parameter:
    """A single tunable parameter.

    ``name`` is ``"<stage>.<attr>"``. ``kind`` is one of ``"float"``,
    ``"int"`` or ``"bool"``. Numeric params carry ``low``/``high`` bounds and a
    perturbation ``step``; boolean params ignore them.
    """

    name: str
    kind: str
    default: Any
    low: float = 0.0
    high: float = 1.0
    step: float = 0.1

    @property
    def stage(self) -> str:
        return self.name.split(".", 1)[0]

    @property
    def attr(self) -> str:
        return self.name.split(".", 1)[1]

    # -- coercion ---------------------------------------------------------

    def coerce(self, value: Any) -> Any:
        """Clamp/round ``value`` to a legal value for this parameter."""
        if self.kind == "bool":
            return bool(value)
        v = float(value)
        v = max(self.low, min(self.high, v))
        if self.kind == "int":
            return int(round(v))
        return round(v, 4)

    # -- sampling ---------------------------------------------------------

    def sample(self, rng: random.Random) -> Any:
        if self.kind == "bool":
            return rng.random() < 0.5
        if self.kind == "int":
            return rng.randint(int(self.low), int(self.high))
        # Sample on the step grid so values stay tidy and comparable.
        n_steps = int(round((self.high - self.low) / self.step))
        if n_steps <= 0:
            return self.coerce(self.low)
        return self.coerce(self.low + self.step * rng.randint(0, n_steps))

    def perturb(self, value: Any, rng: random.Random) -> Any:
        """Return a neighbouring value (one step away, or a bool flip)."""
        if self.kind == "bool":
            return not bool(value)
        cur = float(value)
        direction = rng.choice((-1.0, 1.0))
        # Occasionally take a double step to escape shallow plateaus.
        magnitude = self.step * (2 if rng.random() < 0.25 else 1)
        nxt = self.coerce(cur + direction * magnitude)
        if nxt == self.coerce(cur):
            # Stepped off the edge; go the other way.
            nxt = self.coerce(cur - direction * magnitude)
        return nxt


# ---------------------------------------------------------------------------
# The default space
# ---------------------------------------------------------------------------

_DEFAULT_PARAMETERS: List[Parameter] = [
    # Speaker-reference window (identity / Objective #1).
    Parameter("samples.target_seconds", "float", 10.0, low=6.0, high=14.0, step=1.0),
    Parameter("samples.max_seconds", "float", 15.0, low=10.0, high=20.0, step=1.0),
    # Synthesis timing fit (Objective #3) + conditioning.
    Parameter("generate.duration_tolerance", "float", 0.10, low=0.05, high=0.25, step=0.025),
    Parameter("generate.max_speed", "float", 1.35, low=1.10, high=1.60, step=0.05),
    Parameter("generate.max_fit_iters", "int", 2, low=1, high=4, step=1),
    Parameter("generate.use_clone_prompt", "bool", True),
    # Duration alignment thresholds (timing vs lossy-stretch trade-off).
    Parameter("align.tolerance", "float", 0.10, low=0.05, high=0.25, step=0.025),
    Parameter("align.min_abs_correction_s", "float", 0.12, low=0.05, high=0.30, step=0.025),
]


class ParameterSpace:
    """A named collection of :class:`Parameter` with config-level operators.

    A *config* is a plain ``{name: value}`` dict. The space knows how to
    produce the baseline config, sample a random one, generate neighbours, and
    repair illegal combinations (e.g. ``target_seconds`` above ``max_seconds``).
    """

    def __init__(self, parameters: Iterable[Parameter]):
        self.parameters: List[Parameter] = list(parameters)
        self._by_name: Dict[str, Parameter] = {p.name: p for p in self.parameters}

    # -- access -----------------------------------------------------------

    def __iter__(self):
        return iter(self.parameters)

    def __len__(self) -> int:
        return len(self.parameters)

    def names(self) -> List[str]:
        return [p.name for p in self.parameters]

    def get(self, name: str) -> Parameter:
        return self._by_name[name]

    # -- configs ----------------------------------------------------------

    def default_config(self) -> Dict[str, Any]:
        return {p.name: p.default for p in self.parameters}

    def coerce_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for p in self.parameters:
            if p.name in config:
                out[p.name] = p.coerce(config[p.name])
            else:
                out[p.name] = p.default
        return self.repair(out)

    def repair(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce cross-parameter constraints in place and return ``config``.

        Currently the only constraint is that the reference window's
        ``target_seconds`` must not exceed ``max_seconds`` (otherwise the
        samples stage can never reach its target).
        """
        ts = config.get("samples.target_seconds")
        ms = config.get("samples.max_seconds")
        if ts is not None and ms is not None and ts > ms:
            # Lift the ceiling to the target (cheaper than shrinking the target,
            # and keeps the more-context-is-better bias).
            config["samples.max_seconds"] = self.get("samples.max_seconds").coerce(ts)
        return config

    def random_config(self, rng: random.Random) -> Dict[str, Any]:
        return self.repair({p.name: p.sample(rng) for p in self.parameters})

    def neighbor(
        self, config: Dict[str, Any], rng: random.Random, k: int = 1
    ) -> Dict[str, Any]:
        """Perturb ``k`` randomly-chosen parameters of ``config``."""
        out = dict(config)
        chosen = rng.sample(self.parameters, k=min(k, len(self.parameters)))
        for p in chosen:
            out[p.name] = p.perturb(out.get(p.name, p.default), rng)
        return self.repair(out)


DEFAULT_SPACE = ParameterSpace(_DEFAULT_PARAMETERS)


# ---------------------------------------------------------------------------
# Mapping configs onto the pipeline
# ---------------------------------------------------------------------------


def to_stage_overrides(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Turn a flat ``{"stage.attr": value}`` config into the nested
    ``{stage: {attr: value}}`` map :class:`WorkspacePipeline` expects."""
    overrides: Dict[str, Dict[str, Any]] = {}
    for name, value in config.items():
        stage, _, attr = name.partition(".")
        if not attr:
            continue
        overrides.setdefault(stage, {})[attr] = value
    return overrides


def changed_params(
    config: Dict[str, Any], baseline: Dict[str, Any]
) -> List[str]:
    """Names of parameters whose value differs from ``baseline``."""
    return [k for k, v in config.items() if baseline.get(k) != v]


def earliest_stage(param_names: Iterable[str]) -> Optional[str]:
    """Return the earliest (in pipeline order) stage touched by ``param_names``.

    The evaluator passes this as ``from_stage`` so every stage downstream of a
    changed parameter is re-run — even when a consumer does not declare the
    changed file as an explicit input (e.g. ``generate`` reads speaker profiles
    that ``samples`` produced without declaring them as an input). This is the
    safety net that makes incremental re-runs correct regardless of declared
    I/O gaps.
    """
    stages = {n.split(".", 1)[0] for n in param_names}
    indexed = [(STAGE_ORDER.index(s), s) for s in stages if s in STAGE_ORDER]
    if not indexed:
        return None
    return min(indexed, key=lambda t: t[0])[1]
