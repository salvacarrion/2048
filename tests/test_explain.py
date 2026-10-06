"""The explainable `play` view: strategies that can justify a move record the
score they gave every candidate, and the recorded scores agree with the move
they actually play."""
import pytest

from playbook.evaluation.runner import _format_scores
from playbook.game import SimEnv
from playbook.game.board import Move
from playbook.registry import make_strategy

# Strategies that fill in ``last_scores`` (search + greedy + ntuple). Depths are
# tiny so the test stays fast.
EXPLAINING = {
    "greedy": {},
    "maximization": {"depth": 1},
    "minimax": {"depth": 1},
    "expectimax": {"depth": 1},
    "rollouts": {"depth": 5, "runs": 3},
    "mcts": {"depth": 5, "runs": 8},
    "ntuple": {},
}


@pytest.mark.parametrize("name", sorted(EXPLAINING))
def test_explanation_matches_choice(name):
    env = SimEnv(seed=0)
    board = env.reset()
    legal = env.legal_moves()

    strategy = make_strategy(name, **EXPLAINING[name])
    move = strategy.select_move(board, legal)
    scores = strategy.last_scores

    assert scores is not None, f"{name} should record per-move scores"
    assert set(scores) == set(legal), "one score per legal move, no extras"
    # the move played is a highest-scoring one (ties allowed)
    assert scores[move] == max(scores.values())


def test_random_offers_no_explanation():
    env = SimEnv(seed=0)
    board = env.reset()
    strategy = make_strategy("random")
    strategy.select_move(board, env.legal_moves())
    assert strategy.last_scores is None


def test_format_scores_marks_choice_and_ranks():
    text = _format_scores({Move.UP: 10.0, Move.LEFT: 5.0, Move.RIGHT: 7.0}, Move.UP)
    lines = text.splitlines()
    assert len(lines) == 3
    assert "UP" in lines[0] and "chosen" in lines[0]  # best score listed first
    assert sum("chosen" in ln for ln in lines) == 1   # exactly one marked
    assert "LEFT" in lines[-1]                          # worst score listed last
