"""Search strategy: hill-climbing with epsilon-random exploration.

Why this strategy (vs. brute force / Bayesian / evolutionary):

* The pipeline evaluation is **expensive** (real synthesis), so we want a
  method that improves from the current best with as few evaluations as
  possible — hill-climbing on a small, mostly-separable parameter set does
  exactly that.
* It must be **resumable and stateless between runs**: the next proposal is a
  pure function of the current best configuration plus an RNG, both of which
  are persisted, so a run can stop and continue with no loss.
* A pure greedy climber gets stuck in local optima; mixing in **epsilon-random**
  proposals and **random restarts** after a patience window gives cheap global
  exploration without the bookkeeping of a population method.

Bayesian optimization is the natural next step (noted in the report) but adds a
surrogate-model dependency that is overkill for the first iteration of this
system.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .parameter_space import ParameterSpace


@dataclass
class Proposal:
    """A configuration to evaluate next, tagged with how it was produced."""

    config: Dict[str, Any]
    source: str  # "baseline" | "search" | "random"


class HillClimbSearcher:
    """Propose the next configuration to evaluate.

    The optimizer owns the RNG and the running best; this class is a pure
    proposer so the whole search is deterministic given ``(seed, history)``.
    """

    def __init__(
        self,
        space: ParameterSpace,
        *,
        epsilon: float = 0.2,
        neighbors_k: int = 1,
        restart_patience: int = 8,
    ) -> None:
        self.space = space
        self.epsilon = epsilon
        self.neighbors_k = neighbors_k
        self.restart_patience = restart_patience

    def propose(
        self,
        rng: random.Random,
        *,
        best_config: Optional[Dict[str, Any]],
        stale_steps: int,
        have_baseline: bool,
    ) -> Proposal:
        """Return the next :class:`Proposal`.

        * The very first proposal is the **baseline** (default) config, so every
          run is anchored by a known reference point.
        * Otherwise: a random config with probability ``epsilon`` or after
          ``restart_patience`` non-improving steps (escape a local optimum);
          else a neighbour of the current best (greedy climb).
        """
        if not have_baseline or best_config is None:
            return Proposal(self.space.coerce_config(self.space.default_config()), "baseline")

        if stale_steps >= self.restart_patience or rng.random() < self.epsilon:
            return Proposal(self.space.random_config(rng), "random")

        return Proposal(self.space.neighbor(best_config, rng, k=self.neighbors_k), "search")


# ---------------------------------------------------------------------------
# RNG state (de)serialization for resumable runs
# ---------------------------------------------------------------------------


def rng_to_state(rng: random.Random) -> Any:
    """Return a JSON-serialisable snapshot of ``rng``'s internal state."""
    version, internal, gauss = rng.getstate()
    return [version, list(internal), gauss]


def rng_from_state(state: Any) -> random.Random:
    """Rebuild a :class:`random.Random` from :func:`rng_to_state` output."""
    rng = random.Random()
    version, internal, gauss = state
    rng.setstate((version, tuple(internal), gauss))
    return rng
