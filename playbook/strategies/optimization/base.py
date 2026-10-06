"""The player every optimizer in this family tunes: a weighted sum of features.

A "genome" is a vector of weights over a handful of board features from
:mod:`heuristics <playbook.heuristics>`; the player it defines moves greedily by
``reward + weights · features(afterstate)``. The optimizers (:mod:`genetic`,
:mod:`cmaes`) differ only in *how they propose* weights; they share the player
and the fitness:

  * **Fitness** is the average score over a few *complete* games. Cutting games
    short would reward weights that survive the opening and ignore the late game,
    which is where 2048 is won or lost.
  * **Common random numbers**: within a generation every candidate plays the
    same seeded games, so a lucky spawn sequence cannot decide who survives.
  * Fitness evaluations are independent, so they run in parallel processes.
"""
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from playbook.game.board import Move, simulate_move
from playbook.heuristics import features as F
from playbook.strategies.base import Strategy, Trainable

#: the feature vector a genome weights
FEATURES = [F.gradient, F.max_free, F.total_sum, F.monotonicity, F.potential_merges]
#: fitness games use seeds from here on, far from the benchmark's (0, 1, 2, ...)
TRAIN_SEED = 1_000_000


def _average_score(weights, features, seeds, max_moves):
    """Fitness of one genome: its average score over the games ``seeds`` start.
    Module-level so worker processes can run it."""
    from playbook.evaluation.runner import play_game
    from playbook.game.env import SimEnv
    agent = WeightedFeatureStrategy(weights=weights, features=features)
    return sum(play_game(agent, SimEnv(seed=s), max_moves=max_moves).score
               for s in seeds) / len(seeds)


class WeightedFeatureStrategy(Strategy, Trainable):
    """Greedy player over ``reward + weights · features(afterstate)``.

    Subclasses implement :meth:`train` (the optimizer) on top of :meth:`fitness`.
    """

    name = "weighted-features"

    def __init__(self, weights=None, features=FEATURES, seed=None):
        self.features = list(features)
        if weights is None:
            weights = [1.0] * len(self.features)
        self.weights = np.asarray(weights, dtype=float)
        self.rng = random.Random(seed)
        self._np = np.random.default_rng(seed)

    # -- playing -------------------------------------------------------------
    def _evaluate(self, board):
        return float(sum(w * f(board) for w, f in zip(self.weights, self.features)))

    def select_move(self, board, legal):
        scores = {}
        for move in sorted(legal):
            after, _, reward = simulate_move(board, move)
            scores[Move(move)] = reward + self._evaluate(after)
        self.last_scores = scores
        return max(sorted(scores), key=lambda m: scores[m])

    def observe(self, transition):  # optimizers learn offline, via train()
        pass

    # -- fitness -------------------------------------------------------------
    def fitness(self, genomes, seeds, max_moves=10_000, workers=None):
        """Average score of each genome over the same seeded games (in parallel)."""
        args = [(np.asarray(w, dtype=float), self.features, list(seeds), max_moves)
                for w in genomes]
        if workers == 1:
            return [_average_score(*a) for a in args]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(_average_score, *zip(*args)))

    # -- persistence ---------------------------------------------------------
    def save(self, path):
        np.save(path, self.weights)

    @classmethod
    def load(cls, path, **config):
        return cls(weights=np.load(path), **config)
