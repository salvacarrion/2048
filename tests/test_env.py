"""Environment behavior: reset, step contract, afterstate, determinism."""
import numpy as np

from playbook.game import SimEnv
from playbook.game.board import count_empty


def test_reset_places_two_tiles():
    env = SimEnv(seed=0)
    board = env.reset()
    assert count_empty(board) == 14
    assert env.score == 0


def test_step_returns_contract():
    env = SimEnv(seed=0)
    env.reset()
    move = sorted(env.legal_moves())[0]
    board, reward, done, info = env.step(move)
    assert board.shape == (4, 4)
    assert reward >= 0
    assert isinstance(done, bool)
    assert "afterstate" in info and "legal_moves" in info


def test_seed_is_deterministic():
    a, b = SimEnv(seed=42), SimEnv(seed=42)
    ba, bb = a.reset(), b.reset()
    assert np.array_equal(ba, bb)
    for _ in range(20):
        legal = a.legal_moves()
        if not legal:
            break
        m = sorted(legal)[0]
        ra = a.step(m)
        rb = b.step(m)
        assert np.array_equal(ra[0], rb[0])
        assert ra[1] == rb[1]


def test_illegal_move_is_noop():
    env = SimEnv(seed=1)
    board = env.reset()
    legal = env.legal_moves()
    illegal = [m for m in range(4) if m not in legal]
    if illegal:
        before = board.copy()
        new_board, reward, _, info = env.step(illegal[0])
        assert np.array_equal(new_board, before)
        assert reward == 0.0
        assert info["valid"] is False


def test_game_eventually_terminates():
    import random
    env = SimEnv(seed=3)
    board = env.reset()
    rng = random.Random(3)
    for _ in range(5000):
        legal = env.legal_moves()
        if not legal:
            break
        board, _, done, _ = env.step(rng.choice(sorted(legal)))
        if done:
            break
    assert len(env.legal_moves()) == 0
