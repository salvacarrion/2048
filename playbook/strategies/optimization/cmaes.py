"""CMA-ES: an evolution strategy that learns the *shape* of the search.

Same player and fitness as :mod:`genetic` (see :mod:`base`); only the optimizer
changes. CMA-ES (Covariance Matrix Adaptation Evolution Strategy, Hansen 2001)
samples each generation from a multivariate Gaussian and, from the best samples,
updates three things:

  * the **mean**: moves towards the weighted average of the best candidates;
  * the **step size**: grows while successive steps keep pointing the same way,
    shrinks when they start to cancel out;
  * the **covariance matrix**: stretches the Gaussian along directions that
    keep paying off -- so it discovers, by itself, that features live on very
    different scales and that some weights must move together.

The GA has none of that: a fixed mutation size, the same in every direction.
The algorithm itself comes from the ``cma`` package (``pip install
"playbook-2048[tune]"``); this file is only the glue to our fitness.
"""
import warnings

import numpy as np

from .base import FEATURES, TRAIN_SEED, WeightedFeatureStrategy


def _import_cma():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # it warns when matplotlib is missing; we don't plot
            import cma
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "CMAESStrategy.train needs the cma package. Install with: "
            "pip install 'playbook-2048[tune]'") from exc
    return cma


class CMAESStrategy(WeightedFeatureStrategy):
    name = "cmaes"

    def __init__(self, weights=None, features=FEATURES, seed=None):
        super().__init__(weights=weights, features=features, seed=seed)

    def train(self, env=None, episodes=50, population=16, fitness_games=8, sigma=0.5,
              max_moves=10_000, workers=None, verbose=False, seed=TRAIN_SEED, **kwargs):
        """Run CMA-ES for ``episodes`` generations; return the final mean's fitness
        estimate (the average fitness of the last generation).

        ``env`` is ignored, as in :class:`GeneticStrategy`. The weights kept are the
        distribution's mean: the optimizer's best guess, averaged over many
        samples, rather than the single candidate that got luckiest.
        """
        cma = _import_cma()
        es = cma.CMAEvolutionStrategy(self.weights, sigma,
                                      {"popsize": population, "seed": seed + 1,
                                       "verbose": -9})
        for gen in range(episodes):
            candidates = es.ask()
            seeds = range(seed + gen * fitness_games, seed + (gen + 1) * fitness_games)
            fitness = self.fitness(candidates, seeds, max_moves, workers)
            es.tell(candidates, [-f for f in fitness])        # cma minimizes
            if verbose:
                print(f"  gen {gen + 1:>3}  best {max(fitness):>7.0f}  "
                      f"mean {np.mean(fitness):>7.0f}  step {es.sigma:.3f}  "
                      f"weights {np.round(es.mean, 2)}", flush=True)
        self.weights = np.asarray(es.mean, dtype=float)
        return float(np.mean(fitness))
