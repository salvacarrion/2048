"""Every non-interactive strategy can play a game and the registry resolves them."""
import pytest

from playbook.evaluation import evaluate, play_game
from playbook.game import SimEnv
from playbook.registry import available, make_strategy

# Names that play without a GUI, training, or torch. Search depths kept tiny.
PLAYABLE = {
    "random": {},
    "greedy": {},
    "maximization": {"depth": 1},
    "minimax": {"depth": 1},
    "expectimax": {"depth": 1},
    "mcts": {"depth": 5, "runs": 3},
    "qlearning": {},
    "ntuple": {},
    "genetic": {},
}


def test_registry_lists_all_families():
    names = set(available())
    assert {"random", "greedy", "expectimax", "mcts", "ntuple", "genetic", "dqn"} <= names


@pytest.mark.parametrize("name", sorted(PLAYABLE))
def test_strategy_plays_a_game(name):
    strategy = make_strategy(name, **PLAYABLE[name])
    result = play_game(strategy, SimEnv(seed=0), max_moves=80)
    assert result.moves > 0
    assert result.score >= 0
    assert result.max_tile >= 2


def test_evaluate_aggregates():
    strategy = make_strategy("greedy")
    report = evaluate(strategy, SimEnv, games=3, max_moves=100)
    assert len(report.games) == 3
    assert report.avg_score >= 0
    assert isinstance(report.tile_distribution, dict)


def test_ntuple_learns_something():
    agent = make_strategy("ntuple")
    before = evaluate(agent, SimEnv, games=5, max_moves=500).avg_score
    agent.train(SimEnv(seed=0), episodes=150)
    after = evaluate(agent, SimEnv, games=5, max_moves=2000).avg_score
    assert after > before  # learning should help


def test_ntuple_save_load(tmp_path):
    import numpy as np
    agent = make_strategy("ntuple")
    agent.train(SimEnv(seed=0), episodes=20)
    path = tmp_path / "net.npz"
    agent.save(str(path))
    loaded = make_strategy("ntuple", weights=str(path))
    board = SimEnv(seed=0).reset()
    assert np.isclose(agent.net.value(board), loaded.net.value(board))
