# Optimization / metaheuristics

These strategies **don't play directly**: they search the *parameter space* of a
parametric player and return the best one they find. They are where stochastic,
evolutionary and annealing methods live, and they typically *consume*
[heuristics](../../heuristics/) (a genome is a set of heuristic weights).

| File | Idea |
|---|---|
| [`base.py`](base.py) | the player both optimizers tune: greedy on `reward + weights · features(afterstate)`, plus the shared fitness |
| [`genetic.py`](genetic.py) | `genetic`: evolve a population (elitism, uniform crossover, Gaussian mutation of fixed size) |
| [`cmaes.py`](cmaes.py) | `cmaes`: CMA-ES, an evolution strategy that adapts its step size and the *shape* of its search distribution |

Same player, same fitness, two optimizers: a clean comparison of *how* to search.

```bash
python -m playbook train --strategy genetic --episodes 50 --save genetic.npy   # episodes = generations
python -m playbook train --strategy cmaes   --episodes 50 --save cmaes.npy     # needs: pip install -e ".[tune]"
python -m playbook eval  --strategy cmaes   --weights cmaes.npy --games 50
```

## What the fitness gets right

- **Complete games.** Fitness is the average score over a few full games.
  Cutting games short rewards weights that survive the opening and ignore the
  late game, which is where 2048 is won or lost.
- **Common random numbers.** Within a generation every candidate plays the same
  seeded games, so a lucky spawn sequence cannot decide who survives. (Training
  seeds start at 1,000,000, far from the benchmark's 0, 1, 2, ...)
- **Parallel.** Candidates are scored in separate processes (all CPU cores by
  default), so a generation of 16 candidates × 8 games takes about a minute.

## Results

The bundled `genetic.npy` and `cmaes.npy` come from exactly the commands above
(50 generations × 16 candidates × 8 games, ~20 min each on 16 CPU cores, run at
the same time). On the benchmark's 50 seeded games:

| Player | Avg score | Best | 2048 rate | 4096 rate |
|---|--:|--:|--:|--:|
| untrained (all weights 1) | 4,504 | 9,084 | 0% | 0% |
| `genetic` | 15,263 | 33,688 | 14% | 0% |
| `cmaes` | 17,277 | 47,664 | 24% | 2% |

Both beat `expectimax` and flat `rollouts` in the main benchmark, and `cmaes`
comes within 4% of `mcts` (17,970), while playing ~1,200 moves/s against its 5:
a greedy one-move lookahead with good weights rivals deep search with mediocre
ones. The weights they found, over `[gradient,
max_free, total_sum, monotonicity, potential_merges]`:

```text
genetic  [-0.07  0.20  1.75  1.91   4.73]
cmaes    [-1.28  2.81  3.79  8.07  31.91]
```

Both put the largest weight on `potential_merges` and drop `gradient` (corner
anchoring) to about zero, but CMA-ES scaled the whole vector up ~6x: the player
adds the move's `reward` (in points) to the weighted features, so the overall
scale decides how much the board shape counts against immediate points. With a
fixed mutation size the GA explores that scale slowly; CMA-ES grew its step size
and found it.

**Extend this family** with `annealing.py` (simulated annealing) or
`hillclimb.py`. They all share the same shape: propose parameters → measure
fitness by playing games → keep the good ones. Subclass
`WeightedFeatureStrategy` and only write `train()`.
