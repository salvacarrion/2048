"""Greedy / eager play — take the move with the highest immediate reward."""
import random

from ...game.board import move_reward
from ..base import Strategy


class GreedyStrategy(Strategy):
    """One-step lookahead: choose the legal move that scores the most points now.

    Ties (and the no-information start) are broken randomly. This is the old
    ``EagerSearch``, now comparing true in-game reward instead of a log proxy.
    """

    name = "greedy"

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def select_move(self, board, legal):
        best_move = self.rng.choice(sorted(legal))
        best_reward = -1
        for move in sorted(legal):
            reward = move_reward(board, move)
            if reward > best_reward:
                best_reward = reward
                best_move = move
        return best_move
