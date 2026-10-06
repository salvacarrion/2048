import argparse

import benchmark


def _args(**overrides):
    values = {
        "depth": None,
        "seed": 0,
        "runs": None,
        "rollout_depth": None,
        "ntuple_weights": None,
        "ntuple_untrained": False,
        "lookahead": 0,
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
    cfg = benchmark._config("mcts", _args(seed=123, runs=10, rollout_depth=12))

    assert cfg["seed"] == 123
    assert cfg["runs"] == 10
    assert cfg["depth"] == 12


def test_benchmark_can_request_untrained_ntuple():
    cfg = benchmark._config("ntuple", _args(seed=123, ntuple_untrained=True))

    assert cfg["seed"] == 123
    assert "weights" not in cfg

def test_benchmark_loads_bundled_weights_for_neural_players():
    for name in ("genetic", "cmaes", "qlearning", "ppo", "dqn", "imitation"):
        assert benchmark._config(name, _args())["weights"] == str(benchmark.DEFAULT_WEIGHTS[name])


def test_benchmark_lookahead_applies_to_learned_value_players_only():
    assert benchmark._config("dqn", _args(lookahead=1))["depth"] == 1
    assert "depth" not in benchmark._config("imitation", _args(lookahead=1))
    assert benchmark._label("dqn", {"depth": 1}) == "dqn (depth 1)"
