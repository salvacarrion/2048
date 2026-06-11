"""Board-evaluation functions — the reusable "ideas" about what a good board
looks like. Each takes a board (numpy array of log2 exponents) and returns a
scalar; higher is better. Search and optimization strategies plug these in.
"""
import numpy as np

from ..game.board import col_merges, count_empty, row_merges
from .gradients import GRADIENT_CORNERS, GRADIENT_SNAKE, GRADIENT_TL


def approx_score(board):
    """Approximate the in-game score implied by the tiles on the board."""
    b = np.asarray(board)
    exps = b[b > 1]
    return float(sum((2 ** int(p)) * (int(p) - 1) for p in exps))


def free_tiles(board):
    """Number of empty cells — room to maneuver."""
    return count_empty(board)


def max_tile(board):
    """Largest exponent on the board."""
    return int(np.max(board))


def potential_merges(board):
    """Count of adjacent equal pairs that could merge next."""
    return row_merges(board) + col_merges(board)


def total_sum(board):
    """Sum of all exponents."""
    return int(np.sum(board))


def gradient(board, weights=GRADIENT_TL):
    """Dot the board against a positional weight matrix (corner anchoring)."""
    return float(np.sum(np.multiply(np.asarray(board), weights)))


def gradient_corners(board):
    """Best gradient score over the four corner orientations."""
    return float(max(np.sum(np.multiply(np.asarray(board), g)) for g in GRADIENT_CORNERS))


def gradient_snake(board):
    """Reward a monotonic snake chain across the board."""
    return float(np.sum(np.multiply(np.asarray(board), GRADIENT_SNAKE)))


def max_free(board):
    """Big tiles + breathing room."""
    return free_tiles(board) * max_tile(board)


def monotonicity(board):
    """Reward rows/columns that are monotone (non-increasing or non-decreasing).

    A cleaner replacement for the old ``smoothness`` stub: returns a value in
    roughly [-x, 0], with 0 meaning perfectly monotone everywhere.
    """
    b = np.asarray(board, dtype=np.int64)
    total = 0
    for line in list(b) + list(b.T):
        inc = sum(max(0, line[i + 1] - line[i]) for i in range(len(line) - 1))
        dec = sum(max(0, line[i] - line[i + 1]) for i in range(len(line) - 1))
        total -= min(inc, dec)
    return float(total)
