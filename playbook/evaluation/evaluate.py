"""Run many games and aggregate them into an :class:`EvalReport`."""
import time

from ..game.env import SimEnv
from .report import EvalReport
from .runner import play_game


def evaluate(strategy, env_factory=SimEnv, games=10, max_moves=10_000,
             name=None, render_every=0):
    """Evaluate ``strategy`` over ``games`` games.

    ``env_factory`` is called once per game (a fresh env each time), so this
    works equally for the simulator and the browser. Returns an aggregated
    :class:`EvalReport`.
    """
    results = []
    start = time.time()
    for _ in range(games):
        env = env_factory()
        results.append(play_game(strategy, env, max_moves=max_moves, render_every=render_every))
    return EvalReport(
        strategy=name or getattr(strategy, "name", "strategy"),
        games=results,
        elapsed=time.time() - start,
    )
