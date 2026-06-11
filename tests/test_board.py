"""Golden tests: the refactored board mechanics must reproduce the original
game.py byte for byte. The fixture in fixtures/moves.json was generated from the
pre-refactor implementation (see _gen_golden.py)."""
import json
import os

import numpy as np
import pytest

from playbook.game import Move, is_blocked, legal_moves, make_move, simulate_move

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "moves.json")


def load_cases():
    with open(FIXTURE) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def cases():
    return load_cases()


def test_make_move_matches_reference(cases):
    for case in cases:
        board = np.array(case["board"], dtype=np.int8)
        for move in range(4):
            new_board, points = make_move(board, move)
            expected = np.array(case["moves"][str(move)]["board"], dtype=np.int8)
            assert np.array_equal(new_board, expected), f"board mismatch move={move}"
            assert int(points) == case["moves"][str(move)]["points"], "points mismatch"


def test_legal_moves_matches_reference(cases):
    for case in cases:
        board = np.array(case["board"], dtype=np.int8)
        got = sorted(int(m) for m in legal_moves(board))
        assert got == case["valid_moves"]


def test_is_blocked_matches_reference(cases):
    for case in cases:
        board = np.array(case["board"], dtype=np.int8)
        assert bool(is_blocked(board)) == case["blocked"]


def test_make_move_does_not_mutate_input():
    board = np.array([[1, 1, 0, 0]] * 4, dtype=np.int8)
    before = board.copy()
    make_move(board, Move.LEFT)
    assert np.array_equal(board, before)


def test_reward_is_true_game_score():
    # A row [2,2,2,2] -> two merges into 4 + 4, worth 8 in-game points.
    board = np.zeros((4, 4), dtype=np.int8)
    board[0] = [1, 1, 1, 1]
    new_board, gain_exp, reward = simulate_move(board, Move.LEFT)
    assert reward == 8           # 4 + 4 (true value points)
    assert gain_exp == 4         # 2 + 2 (legacy exponent points)
    assert list(new_board[0]) == [2, 2, 0, 0]
