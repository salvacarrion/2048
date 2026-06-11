"""One-shot generator for the golden fixture.

Runs the ORIGINAL game.py (repo root) over a set of seeded random boards and
snapshots make_move / get_valid_moves / is_blocked outputs. The new package's
board mechanics are validated against this fixture so the refactor provably
preserves behavior. This script is historical: it stops working once the old
game.py is removed, which is fine — the fixture (moves.json) is what matters.
"""
import json
import os
import sys

import numpy as np

# Import the original, pre-refactor implementation from the repo root.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import game as old  # noqa: E402


def random_board(rng):
    """A 4x4 board of log2 exponents with mixed density (0=empty)."""
    density = rng.uniform(0.2, 0.95)
    board = np.zeros((4, 4), dtype=np.int8)
    for i in range(4):
        for j in range(4):
            if rng.random() < density:
                board[i][j] = rng.integers(1, 12)  # exponents 1..11 (2..2048)
    return board


def main():
    rng = np.random.default_rng(2048)
    cases = []
    for _ in range(400):
        board = random_board(rng)
        entry = {
            "board": board.tolist(),
            "valid_moves": sorted(old.get_valid_moves(board)),
            "blocked": bool(old.is_blocked(board)),
            "moves": {},
        }
        for move in range(4):
            new_board, points = old.make_move(board, move)
            entry["moves"][str(move)] = {
                "board": np.asarray(new_board).tolist(),
                "points": int(points),
            }
        cases.append(entry)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "moves.json")
    with open(path, "w") as f:
        json.dump(cases, f)
    print(f"Wrote {len(cases)} cases to {path}")


if __name__ == "__main__":
    main()
