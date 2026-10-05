# Deep reinforcement learning

The neural-network chapter. Same `Strategy` interface as everything else, but the
value function is a net (see [`networks.py`](../../networks.py), shared with
supervised imitation) instead of tables. torch is imported **lazily**, so the
rest of the package runs without it:

```bash
pip install "playbook-2048[deep]"     # installs torch
```

| File | Idea | Status |
|---|---|---|
| `dqn.py` | Deep Q-Network on afterstates: replay buffer, target network, n-step returns | implemented + trained weights (`dqn.pt`) |

## What makes it work on 2048

A textbook DQN (a net mapping a board to four Q-values, 1-step targets) learns
very slowly here. Two changes fix most of that, and both are worth reading in
[`dqn.py`](dqn.py):

1. **Learn `V(afterstate)`, not `Q(s, ·)`.** The slide is deterministic and we
   can simulate it, so `Q(s, a) = r(s, a) + V(afterstate(s, a))` and the net
   only has to learn the value of the board *after* our move, before the random
   tile. This is the same trick as the n-tuple network.
2. **n-step returns.** With a 1-step target, value information moves back one
   move per target-network refresh. Games last thousands of moves, so the net
   ends up predicting ~20 points from an opening board that is worth thousands,
   and plays like a greedy heuristic. Summing the points actually scored over
   the next `n` moves before bootstrapping (`n_step=10`) fixes that: the training
   log prints `pred` (what the net expects to score from an opening board) next
   to `avg` (what it actually scores), so you can watch the two meet.

Training feeds the GPU with hundreds of games played at once on the batched
simulator ([`game/vector.py`](../../../../game/vector.py)): a single Python
`SimEnv` manages ~3k moves/s, far too few to keep a GPU busy.

## Train and play

The bundled `dqn.pt` was trained in two stages on a GTX 1070 (MLP value net,
0.8M weights, 256 games in parallel):

```bash
# 1) ~2 h, ~85k games: learns until it plateaus (~35k points, 2048 in ~72% of games)
python -m playbook train --strategy dqn --episodes 85000 --save dqn.pt
# 2) ~12 min, a few thousand games at a 5x lower learning rate: +6k points, 4096 in ~half the games
python -m playbook train --strategy dqn --weights dqn.pt --episodes 6000 --lr 2e-5 --save dqn.pt
```

Training keeps (and saves) the network whose games averaged the best score, so
you can stop it at any time. Then play:

```bash
python -m playbook eval --strategy dqn --weights dqn.pt --games 50            # greedy
python -m playbook eval --strategy dqn --weights dqn.pt --games 50 --depth 1  # + expectimax
```

| `dqn.pt` on fresh games | avg score | reaches 2048 | reaches 4096 | reaches 8192 |
|---|--:|--:|--:|--:|
| greedy (`--depth 0`, 300 games) | 41,229 | 79% | 48% | 0% |
| one level of expectimax (`--depth 1`, 200 games) | 80,402 | 99% | 95% | 14% |

`--depth 1` searches one spawn ahead with the learned value at the leaves
([`search/lookahead.py`](../../../search/lookahead.py)): every leaf of the tree
is scored in a single batched forward pass, so it stays fast.

`policy_gradient.py` (REINFORCE / actor-critic) is a natural next file to add.
