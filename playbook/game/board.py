"""Board mechanics for 2048.

Boards are 4x4 numpy arrays of **log2 exponents**, not raw tile values:
``0`` means empty, ``1`` means the tile 2, ``2`` means 4, ... ``11`` means 2048.
This compact encoding keeps every heuristic and the spawn logic simple, and it
is shared by the whole package.

Moves are encoded UDLR everywhere in Python (``Move.UP == 0`` ... ``RIGHT == 3``).
Only the browser controller translates to the game's internal order.

The merge logic lives in a single 1-D helper, :func:`_collapse_left_line`, and
every direction is expressed by orienting the board so the move becomes a
leftward collapse. This is intentionally one small function instead of four
near-duplicate ones, so the rules are easy to read and test.
"""
from copy import deepcopy
from enum import IntEnum

import numpy as np

SIZE = 4


class Move(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


MOVE_NAMES = {Move.UP: "UP", Move.DOWN: "DOWN", Move.LEFT: "LEFT", Move.RIGHT: "RIGHT"}


def move_name(move):
    return MOVE_NAMES[Move(move)]


# --------------------------------------------------------------------------- #
# Construction & conversion
# --------------------------------------------------------------------------- #
def empty_board():
    return np.zeros((SIZE, SIZE), dtype=np.int8)


def to_value(exp):
    """Exponent -> displayed tile value (0 stays 0)."""
    return 0 if exp == 0 else 2 ** int(exp)


def to_values(board):
    """Board of exponents -> nested list of displayed tile values."""
    return [[to_value(c) for c in row] for row in board]


def tile_score(exp):
    """In-game points a tile of this exponent is worth when created."""
    return 0 if exp <= 1 else (int(exp) - 1) * (2 ** int(exp))


# --------------------------------------------------------------------------- #
# Core slide + merge
# --------------------------------------------------------------------------- #
def _collapse_left_line(line):
    """Collapse a single row of 4 exponents to the left (standard 2048 rules).

    Returns ``(new_line, gain_exp, reward)`` where ``gain_exp`` is the sum of the
    *exponents* produced by merges (the legacy score signal, kept so existing
    heuristics behave identically) and ``reward`` is the true in-game score
    delta (the sum of merged tile *values*).
    """
    tiles = [int(x) for x in line if x != 0]   # plain ints: avoid int8 overflow in 1<<merged
    out, gain_exp, reward = [], 0, 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            merged = tiles[i] + 1            # new exponent
            out.append(merged)
            gain_exp += merged
            reward += 1 << merged            # 2 ** merged == true tile value
            i += 2
        else:
            out.append(tiles[i])
            i += 1
    out += [0] * (SIZE - len(out))
    return out, gain_exp, reward


# For each move, a pair of (view, unview) so the move becomes a leftward
# collapse: orient the board, collapse every row, then orient back.
_ORIENT = {
    Move.UP:    (lambda b: b.T,          lambda o: o.T),
    Move.DOWN:  (lambda b: b.T[:, ::-1], lambda o: o[:, ::-1].T),
    Move.LEFT:  (lambda b: b,            lambda o: o),
    Move.RIGHT: (lambda b: b[:, ::-1],   lambda o: o[:, ::-1]),
}


def simulate_move(board, move):
    """Apply a slide+merge without spawning a tile.

    Returns ``(new_board, gain_exp, reward)``. ``new_board`` is a fresh array
    (the input is never mutated). For an illegal move ``new_board`` equals the
    input board and both scores are 0.
    """
    view, unview = _ORIENT[Move(move)]
    work = view(np.asarray(board, dtype=np.int8))
    out = np.zeros_like(work)
    gain_exp = reward = 0
    for r in range(SIZE):
        new_line, g, rw = _collapse_left_line(list(work[r]))
        out[r] = new_line
        gain_exp += g
        reward += rw
    return np.ascontiguousarray(unview(out)), gain_exp, reward


def make_move(board, move):
    """Slide+merge, returning ``(new_board, gain_exp)`` (legacy exponent score)."""
    new_board, gain_exp, _ = simulate_move(board, move)
    return new_board, gain_exp


def move_reward(board, move):
    """True in-game score gained by ``move`` (sum of merged tile values)."""
    _, _, reward = simulate_move(board, move)
    return reward


# --------------------------------------------------------------------------- #
# Empty cells & tile spawning
# --------------------------------------------------------------------------- #
def free_cells(board):
    return [(i, j) for i in range(SIZE) for j in range(SIZE) if board[i][j] == 0]


def count_empty(board):
    return int(np.count_nonzero(np.asarray(board) == 0))


def set_tile(board, i, j, exp, copy=True):
    """Return a board with cell (i, j) set to exponent ``exp``."""
    nb = deepcopy(board) if copy else board
    nb[i][j] = exp
    return nb


def add_random_tile(board, rng):
    """Spawn a tile in place: exponent 1 (value 2) at 90%, exponent 2 (4) at 10%."""
    cells = free_cells(board)
    if not cells:
        return board
    i, j = cells[rng.randrange(len(cells))]
    board[i][j] = 1 if rng.random() < 0.9 else 2
    return board


# --------------------------------------------------------------------------- #
# Merge-availability helpers (used by heuristics)
# --------------------------------------------------------------------------- #
def row_merges(board):
    """Count of adjacent equal non-zero pairs within each row."""
    b = np.asarray(board)
    return int(sum(
        1 for i in range(SIZE) for j in range(SIZE - 1)
        if b[i][j] != 0 and b[i][j] == b[i][j + 1]
    ))


def col_merges(board):
    """Count of adjacent equal non-zero pairs within each column."""
    b = np.asarray(board)
    return int(sum(
        1 for j in range(SIZE) for i in range(SIZE - 1)
        if b[i][j] != 0 and b[i][j] == b[i + 1][j]
    ))


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render(board):
    """Return a printable string of the board in displayed tile values."""
    lines = []
    for row in to_values(board):
        lines.append(" ".join("%6d" % c for c in row))
    return "\n".join(lines)
