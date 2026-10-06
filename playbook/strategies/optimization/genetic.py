"""Genetic algorithm that tunes a heuristic player.

A metaheuristic does not play directly — it searches the *parameter space* of a
player. Here a "genome" is a vector of weights over a handful of board features;
the player it defines moves greedily by the weighted sum (see :mod:`base`). The
GA evolves the weights, using the average game score as fitness. This is the
example of a *stochastic / evolutionary* strategy that consumes :mod:`heuristics
<playbook.heuristics>`.

Each generation: score every genome on the same games, keep the ``elite`` best
unchanged, and fill the rest of the population with children of two random
elite parents (uniform crossover + Gaussian mutation).
"""
import numpy as np

from .base import FEATURES, TRAIN_SEED, WeightedFeatureStrategy


class GeneticStrategy(WeightedFeatureStrategy):
    name = "genetic"

    def __init__(self, weights=None, features=FEATURES, seed=None):
        super().__init__(weights=weights, features=features, seed=seed)

    def train(self, env=None, episodes=50, population=16, fitness_games=8, elite=4,
              sigma=0.5, max_moves=10_000, workers=None, verbose=False, seed=TRAIN_SEED, **kwargs):
        """Evolve the weights for ``episodes`` generations; return the best fitness.

        ``env`` is ignored: fresh seeded simulators measure fitness (generation
        ``g`` plays seeds ``seed + g * fitness_games ...``). The weights kept are
        the best genome of the last generation -- elites survive unchanged, so it
        has done well on several different sets of games, not one lucky set.
        """
        dim = len(self.features)
        pop = [self.weights + self._np.normal(0, sigma, dim) for _ in range(population)]
        pop[0] = self.weights.copy()  # keep the seed genome

        for gen in range(episodes):
            seeds = range(seed + gen * fitness_games, seed + (gen + 1) * fitness_games)
            fitness = self.fitness(pop, seeds, max_moves, workers)
            order = np.argsort(fitness)[::-1]
            if verbose:
                print(f"  gen {gen + 1:>3}  best {fitness[order[0]]:>7.0f}  "
                      f"mean {np.mean(fitness):>7.0f}  weights {np.round(pop[order[0]], 2)}",
                      flush=True)
            parents = [pop[i] for i in order[:elite]]
            if gen == episodes - 1:
                break
            children = [p.copy() for p in parents]
            while len(children) < population:
                a, b = self.rng.sample(parents, 2)
                mask = self._np.random(dim) < 0.5
                children.append(np.where(mask, a, b) + self._np.normal(0, sigma, dim))
            pop = children

        self.weights = parents[0]
        return float(fitness[order[0]])
