"""Genetic algorithm that tunes a heuristic player.

A metaheuristic does not play directly — it searches the *parameter space* of a
player. Here a "genome" is a vector of weights over a handful of board features;
the player it defines moves greedily by the weighted sum. The GA evolves the
weights, using the average game score as fitness. This is the example of a
*stochastic / evolutionary* strategy that consumes :mod:`heuristics
<playbook.heuristics>`.
"""
import random

import numpy as np

from playbook.game.board import simulate_move
from playbook.heuristics import features as F
from playbook.strategies.base import Strategy, Trainable

#: the feature vector a genome weights
FEATURES = [F.gradient, F.max_free, F.total_sum, F.monotonicity, F.potential_merges]


class GeneticStrategy(Strategy, Trainable):
    name = "genetic"

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
        best_move, best_val = None, float("-inf")
        for move in sorted(legal):
            after, _, reward = simulate_move(board, move)
            val = reward + self._evaluate(after)
            if best_move is None or val > best_val:
                best_move, best_val = move, val
        return best_move

    def observe(self, transition):  # GA learns offline, via train()
        pass

    # -- evolution -----------------------------------------------------------
    def _fitness(self, weights, env_factory, games, max_moves):
        from playbook.evaluation.runner import play_game
        agent = GeneticStrategy(weights=weights, features=self.features)
        return sum(play_game(agent, env_factory(), max_moves=max_moves).score
                   for _ in range(games)) / games

    def train(self, env, episodes=20, population=12, fitness_games=2,
              max_moves=400, sigma=0.5, elite=3, verbose=False):
        """Evolve the weights. ``episodes`` is the number of generations.

        ``env`` is ignored — fresh simulators are spun up to measure fitness.
        The best genome found becomes this strategy's weights.
        """
        from playbook.game.env import SimEnv
        env_factory = SimEnv

        dim = len(self.features)
        pop = [self.weights + self._np.normal(0, sigma, dim) for _ in range(population)]
        pop[0] = self.weights.copy()  # keep the seed genome

        best_w, best_f = self.weights.copy(), float("-inf")
        for gen in range(1, episodes + 1):
            scored = sorted(
                ((self._fitness(w, env_factory, fitness_games, max_moves), w) for w in pop),
                key=lambda t: t[0], reverse=True,
            )
            if scored[0][0] > best_f:
                best_f, best_w = scored[0][0], scored[0][1].copy()
            if verbose:
                print(f"  gen {gen:>3}  best fitness: {scored[0][0]:.0f}")

            parents = [w for _, w in scored[:elite]]
            children = list(parents)
            while len(children) < population:
                a, b = self.rng.sample(parents, 2)
                mask = self._np.random(dim) < 0.5
                child = np.where(mask, a, b) + self._np.normal(0, sigma, dim)
                children.append(child)
            pop = children

        self.weights = best_w
        return best_f

    # -- persistence ---------------------------------------------------------
    def save(self, path):
        np.save(path, self.weights)

    @classmethod
    def load(cls, path):
        return cls(weights=np.load(path))
