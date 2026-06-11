# Search strategies

Players that **look ahead in the game tree** and score the boards they reach
with a [heuristic](../../heuristics/). They share a model of the game (the
simulator) and differ only in how they treat the random tile spawn.

| Strategy | How it treats the spawn | Notes |
|---|---|---|
| `maximization` | one sampled random tile between plies | simplest lookahead |
| `minimax` | an **adversary** placing the worst tile (α-β pruned) | pessimistic |
| `expectimax` | **chance**: averages over every empty cell × {2 @ 0.9, 4 @ 0.1} | the principled choice for 2048 |
| `mcts` | random rollouts (`runs` per move, length `depth`) | Monte-Carlo, no heuristic of the tree, only of the leaves |

All inject their heuristic, so you can swap evaluation ideas without touching the
search:

```python
from playbook.strategies.search import ExpectimaxStrategy
from playbook.heuristics import get_heuristic
ExpectimaxStrategy(heuristic=get_heuristic("gradient"), depth=3)
```

**Deviations from the original repo (made for correctness/clarity):**
- `expectimax` now spawns a 2 (exp 1) at 90% and a 4 (exp 2) at 10% — the real
  distribution. The original mistakenly spawned exponents 2 and 4.
- `minimax` drops a deterministic 2 in the worst cell instead of a random tile,
  so the search is reproducible.
