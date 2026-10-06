# Search strategies

Players that **look ahead in the game tree** and score the boards they reach
with a [heuristic](../../heuristics/). They share a model of the game (the
simulator) and differ only in how they treat the random tile spawn.

| Strategy | How it treats the spawn | Notes |
|---|---|---|
| `maximization` | one sampled random tile between plies | simplest lookahead |
| `minimax` | an **adversary** placing the worst tile (α-β pruned) | pessimistic |
| `expectimax` | **chance**: averages over every empty cell × {2 @ 0.9, 4 @ 0.1} | the principled choice for 2048 |
| `rollouts` | random playouts (`runs` per move, length `depth`), averaged | "flat" Monte Carlo: every move gets the same playouts; each scored by the points it earns + the leaf heuristic |
| `mcts` | random playouts too, but spent **adaptively** by a search tree | Monte Carlo Tree Search (UCT) with chance nodes for the spawn; `runs` simulations per move |

All inject their heuristic, so you can swap evaluation ideas without touching the
search:

```python
from playbook.strategies.search import ExpectimaxStrategy
from playbook.heuristics import get_heuristic
ExpectimaxStrategy(heuristic=get_heuristic("gradient"), depth=3)
```

[`lookahead.py`](lookahead.py) is the same expectimax idea for the *learning*
players: instead of a hand-written heuristic at the leaves it uses a learned
`V(afterstate)`, and it expands the tree level by level on numpy batches so a
neural network scores every leaf in one forward pass. It is what `--depth` means
for `ntuple` and `dqn` (`dqn --depth 1` reaches 4096 in ~95% of games).

`rollouts` vs `mcts` is the cleanest comparison in the catalog: same playouts,
same scoring, same budget (`mcts --runs 80` ≈ `rollouts --runs 20` × ~4 moves).
The only difference is that MCTS keeps a tree, steers new playouts toward the
moves that have done well (UCB1: `mean + c·sqrt(ln N / n)`, explore vs exploit),
and refines the decisions below the root as well. Its `--explain` view shows the
share of simulations each move received. On the benchmark's 50 seeded games:
`rollouts` 12,406 points on average (2048 in 6% of games), `mcts` 17,970 (26%),
at the same ~5 moves/s.

**Deviations from the original repo (made for correctness/clarity):**
- `expectimax` now spawns a 2 (exp 1) at 90% and a 4 (exp 2) at 10% — the real
  distribution. The original mistakenly spawned exponents 2 and 4.
- `minimax` drops a deterministic 2 in the worst cell instead of a random tile,
  so the search is reproducible.
- `rollouts` (called `mcts` in earlier versions, although it builds no tree)
  scores a playout by the **points it earns** along the way (plus the leaf
  heuristic), not by the heuristic of the final board alone. A long random tail
  always ends in a near-dead board, so its heuristic barely depends on the first
  move; the realized score does. This is what lifts it from ~greedy level to
  comfortably above the other search players.
