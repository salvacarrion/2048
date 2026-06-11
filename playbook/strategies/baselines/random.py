"""Random play — the absolute baseline."""
import random

from ..base import Strategy


class RandomStrategy(Strategy):
    """Pick uniformly at random among the legal moves."""

    name = "random"

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def select_move(self, board, legal):
        return self.rng.choice(sorted(legal))
