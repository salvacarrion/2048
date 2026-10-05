"""Every non-interactive strategy can play a game and the registry resolves them."""
import numpy as np
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


def test_qlearning_learns_and_round_trips(tmp_path):
    agent = make_strategy("qlearning", seed=0)
    agent.train(SimEnv(seed=0), episodes=30)
    assert len(agent.q) > 0
    path = tmp_path / "q.pkl"
    agent.save(str(path))
    loaded = make_strategy("qlearning", weights=str(path))
    assert loaded.q == agent.q
    assert play_game(loaded, SimEnv(seed=0), max_moves=50).moves > 0


def test_dqn_trains_plays_and_round_trips(tmp_path):
    pytest.importorskip("torch")
    agent = make_strategy("dqn", device="cpu", seed=0, channels=8, hidden=16)
    agent.train(episodes=4, n_envs=4, batch_size=32, learn_start=64, target_every=10)
    path = tmp_path / "dqn.pt"
    agent.save(str(path))
    for depth in (0, 1):
        loaded = make_strategy("dqn", weights=str(path), device="cpu", depth=depth)
        result = play_game(loaded, SimEnv(seed=0), max_moves=30)
        assert result.moves > 0 and loaded.last_scores
    board = SimEnv(seed=0).reset()
    assert np.allclose(agent.move_values(board[None]), loaded.move_values(board[None], depth=0))


def test_imitation_copies_its_teacher(tmp_path):
    pytest.importorskip("torch")
    teacher = make_strategy("greedy")
    agent = make_strategy("imitation", device="cpu", seed=0, channels=8, hidden=32)
    boards, moves = agent.train(episodes=8, teacher=teacher, rounds=2, epochs=3, n_envs=4)
    assert len(boards) == len(moves) > 0
    path = tmp_path / "imitation.pt"
    agent.save(str(path))
    loaded = make_strategy("imitation", weights=str(path), device="cpu")
    assert play_game(loaded, SimEnv(seed=0), max_moves=30).moves > 0


def test_imitation_soft_targets_from_a_scoring_teacher():
    from benchmark import DEFAULT_NTUPLE_WEIGHTS
    from playbook.game.vector import VecSimEnv, expand
    from playbook.strategies.learning.supervised.imitation import teacher_targets
    teacher = make_strategy("ntuple", weights=str(DEFAULT_NTUPLE_WEIGHTS))
    boards = VecSimEnv(64, seed=0).boards
    moves, targets = teacher_targets(teacher, boards, temperature=50.0)
    _, _, legal = expand(boards)
    assert np.allclose(targets.sum(axis=1), 1.0)
    assert (targets[~legal] == 0).all()
    assert np.array_equal(targets.argmax(axis=1), moves)
