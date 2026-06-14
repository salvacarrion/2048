import argparse

import benchmark


def _args(**overrides):
    values = {
        "depth": None,
        "seed": 0,
        "mcts_runs": None,
        "mcts_depth": None,
        "ntuple_weights": None,
        "ntuple_untrained": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_benchmark_seeds_stochastic_strategy_profiles():
    assert benchmark._config("random", _args(seed=123))["seed"] == 123

    cfg = benchmark._config("expectimax", _args(seed=123, depth=4))
    assert cfg["seed"] == 123
    assert cfg["depth"] == 4


def test_benchmark_uses_bundled_ntuple_weights_by_default():
    cfg = benchmark._config("ntuple", _args(seed=123))

    assert cfg["seed"] == 123
    assert cfg["weights"] == str(benchmark.DEFAULT_NTUPLE_WEIGHTS)


def test_benchmark_can_override_mcts_profile():
    cfg = benchmark._config("mcts", _args(seed=123, mcts_runs=10, mcts_depth=12))

    assert cfg["seed"] == 123
    assert cfg["runs"] == 10
    assert cfg["depth"] == 12


def test_benchmark_can_request_untrained_ntuple():
    cfg = benchmark._config("ntuple", _args(seed=123, ntuple_untrained=True))

    assert cfg["seed"] == 123
    assert "weights" not in cfg