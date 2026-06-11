"""Play a single game of one strategy against one environment."""
import time

import numpy as np

from ..game.board import move_name, render, to_values
from .report import GameResult


def play_game(strategy, env, max_moves=10_000, render_every=0):
    """Run one game and return a :class:`GameResult`.

    Works with any :class:`~playbook.game.env.Env` (simulator or browser), so
    the same loop evaluates a player offline or live.
    """
    board = env.reset()
    strategy.reset()
    start = time.time()
    moves = 0

    while moves < max_moves:
        legal = env.legal_moves()
        if not legal:
            break
        move = strategy.select_move(board, legal)
        board, _, done, _ = env.step(move)
        moves += 1

        if render_every and moves % render_every == 0:
            print(f"move #{moves} [{move_name(move)}]  score={env.score}")
            print(render(board), "\n")
        if done:
            break

    max_tile = int(np.max(to_values(board)))
    return GameResult(
        score=int(env.score),
        max_tile=max_tile,
        moves=moves,
        elapsed=time.time() - start,
    )
