"""Tests for the optimization parameter space."""
import random

import pytest

from src.optimization.parameter_space import (
    DEFAULT_SPACE,
    Parameter,
    ParameterSpace,
    changed_params,
    earliest_stage,
    to_stage_overrides,
)


def test_default_config_has_every_parameter():
    cfg = DEFAULT_SPACE.default_config()
    assert set(cfg) == set(DEFAULT_SPACE.names())


def test_coerce_clamps_and_types():
    p = Parameter("generate.max_speed", "float", 1.35, low=1.1, high=1.6, step=0.05)
    assert p.coerce(99) == 1.6
    assert p.coerce(0) == 1.1
    pi = Parameter("generate.max_fit_iters", "int", 2, low=1, high=4, step=1)
    assert pi.coerce(3.4) == 3 and isinstance(pi.coerce(3.4), int)
    pb = Parameter("generate.use_clone_prompt", "bool", True)
    assert pb.coerce(0) is False


def test_random_config_is_legal_and_deterministic():
    a = DEFAULT_SPACE.random_config(random.Random(7))
    b = DEFAULT_SPACE.random_config(random.Random(7))
    assert a == b  # deterministic given seed
    for name, val in a.items():
        p = DEFAULT_SPACE.get(name)
        if p.kind != "bool":
            assert p.low <= float(val) <= p.high


def test_neighbor_changes_at_least_one_param():
    base = DEFAULT_SPACE.default_config()
    nb = DEFAULT_SPACE.neighbor(base, random.Random(3), k=1)
    assert nb != base
    assert len(changed_params(nb, base)) >= 1


def test_repair_enforces_target_le_max():
    cfg = DEFAULT_SPACE.coerce_config(
        {"samples.target_seconds": 14, "samples.max_seconds": 10}
    )
    assert cfg["samples.max_seconds"] >= cfg["samples.target_seconds"]


def test_to_stage_overrides_splits_names():
    ov = to_stage_overrides({"generate.max_speed": 1.4, "align.tolerance": 0.2})
    assert ov == {"generate": {"max_speed": 1.4}, "align": {"tolerance": 0.2}}


def test_earliest_stage_picks_pipeline_order():
    assert earliest_stage(["align.tolerance", "generate.max_speed"]) == "generate"
    assert earliest_stage(["align.tolerance", "samples.target_seconds"]) == "samples"
    assert earliest_stage([]) is None


def test_perturb_bool_flips():
    p = Parameter("x.flag", "bool", True)
    assert p.perturb(True, random.Random(0)) is False
