"""Play a single game of one strategy against one environment."""
import time

import numpy as np

from ..game.board import move_name, render, to_values
from .report import GameResult


def play_game(strategy, env, max_moves=10_000, render_every=0,
              delay=0.0, step=False, explain=True):
    """Run one game and return a :class:`GameResult`.

    Works with any :class:`~playbook.game.env.Env` (simulator or browser), so
    the same loop evaluates a player offline or live.

    The watch-and-learn knobs only matter when ``render_every`` is set (i.e. when
    you are watching, as in the ``play`` command): ``delay`` sleeps that many
    seconds between rendered moves, ``step`` instead waits for Enter before each
    one, and ``explain`` prints the score the strategy gave every candidate move
    — the *why* behind its choice (see :attr:`Strategy.last_scores`).
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
        scores = strategy.last_scores
        board, _, done, _ = env.step(move)
        moves += 1

        if render_every and moves % render_every == 0:
            print(f"move #{moves} [{move_name(move)}]  score={env.score}")
            if explain and scores:
                print(_format_scores(scores, move))
            print(render(board), "\n")
            _pace(delay, step)
        if done:
            break

    max_tile = int(np.max(to_values(board)))
    return GameResult(
        score=int(env.score),
        max_tile=max_tile,
        moves=moves,
        elapsed=time.time() - start,
    )


def _format_scores(scores, chosen):
    """Render ``{Move: score}`` as one aligned line per move, best first, marking
    the move that was played — the per-turn 'why' shown in the live view."""
    lines = []
    for move, value in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        mark = "  <-- chosen" if move == chosen else ""
        lines.append(f"    {move_name(move):>5}{value:14.1f}{mark}")
    return "\n".join(lines)


def _pace(delay, step):
    """Throttle the live view: wait for Enter (``step``) or sleep ``delay`` s."""
    if step:
        try:
            input("    ...press Enter for the next move (Ctrl-C to stop) ")
        except EOFError:
            pass
    elif delay > 0:
        time.sleep(delay)
