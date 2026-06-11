# Optimization / metaheuristics

These strategies **don't play directly** — they search the *parameter space* of a
parametric player and return the best one they find. They are where stochastic,
evolutionary and annealing methods live, and they typically *consume*
[heuristics](../../heuristics/) (a genome is a set of heuristic weights).

| Strategy | Idea |
|---|---|
| `genetic` | evolve a population of weight vectors; fitness = average game score |

Run an evolution and then play with the winning weights:

```python
from playbook.strategies.optimization import GeneticStrategy
g = GeneticStrategy(seed=0)
g.train(env=None, episodes=30, population=16, fitness_games=3)   # episodes = generations
g.select_move(board, legal)        # now plays with the evolved weights
```

**Extend this family** with `evolution.py` (CMA-ES — there is an optional `tune`
extra), `annealing.py` (simulated annealing) or `hillclimb.py`. They all share
the same shape: propose parameters → measure fitness by playing games → keep the
good ones.
