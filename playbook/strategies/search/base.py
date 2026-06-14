"""Shared scaffolding for tree-search strategies.

A search strategy looks ahead in the game tree and scores leaf boards with a
:mod:`heuristic <playbook.heuristics>`. It does not know *which* heuristic — that
is injected, so you can swap evaluation ideas without touching the search.
"""
import random

from ...heuristics.combined import default_heuristic
from ..base import Strategy

NEG_INF = float("-inf")
POS_INF = float("inf")


class SearchStrategy(Strategy):
    def __init__(self, heuristic=None, depth=3, seed=None):
        self.heuristic = heuristic if heuristic is not None else default_heuristic()
        self.depth = depth
        self.rng = random.Random(seed)

    def _pick(self, scores):
        """Record the per-move ``{Move: score}`` (for the explainable ``play``
        view) and return the best move. Ties break toward the lower move index,
        matching the original deterministic selection.
        """
        self.last_scores = scores
        best_move, best = None, NEG_INF
        for move in sorted(scores):
            if best_move is None or scores[move] > best:
                best_move, best = move, scores[move]
        return best_move
