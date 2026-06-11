"""Game-rule queries derived from the board mechanics.

These answer "what can happen from here" questions that both the environment
and the search strategies rely on: which moves are legal, and whether the game
is over.
"""
import numpy as np

from .board import SIZE, Move, simulate_move


def legal_moves(board):
    """Set of moves that actually change the board."""
    board = np.asarray(board)
    moves = set()
    for move in Move:
        new_board, _, _ = simulate_move(board, move)
        if not np.array_equal(new_board, board):
            moves.add(Move(move))
    return moves


def is_terminal(board):
    """True when no move changes the board (the game is over)."""
    return len(legal_moves(board)) == 0


def is_blocked(board):
    """Loose "stuck" test used to bottom out search.

    Returns True only when no two horizontally or vertically adjacent cells are
    equal. Two empty cells count as equal, so a board with adjacent blanks is
    reported as not blocked. Kept as a fast inner-loop check for the tree
    searches; use :func:`is_terminal` for a precise game-over test.
    """
    b = np.asarray(board)
    for i in range(SIZE):
        for j in range(SIZE):
            if j + 1 < SIZE and b[i][j] == b[i][j + 1]:
                return False
            if i + 1 < SIZE and b[i][j] == b[i + 1][j]:
                return False
    return True
