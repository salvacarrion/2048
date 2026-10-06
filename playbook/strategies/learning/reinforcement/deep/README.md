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
| `dqn.py` | Deep Q-Network on afterstates: replay buffer, target network, n-step returns (value-based) | implemented + trained weights (`dqn.pt`) |
| `ppo.py` | Proximal Policy Optimization: actor-critic, GAE, clipped updates (policy-based) | implemented + trained weights (`ppo.pt`) |

The two families of deep RL, side by side: `dqn` learns how good each board is
and plays the best-valued move; `ppo` learns the probabilities of the moves
directly and plays the most likely one.

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

## PPO: learning the policy itself

[`ppo.py`](ppo.py) is the policy-gradient counterpart: one network outputs four
move probabilities (illegal moves masked out) plus a value `V(s)`. It plays a
batch of games by *sampling* from its own policy, measures how much better than
expected each move turned out (the advantage, with GAE), and nudges the
probabilities accordingly, never more than `clip` (20%) per update. The data is
used for a few epochs and thrown away: PPO is on-policy, it cannot learn from a
replay buffer of older policies the way DQN does.

```bash
python -m playbook train --strategy ppo --episodes 200000 --save ppo.pt   # GPU
python -m playbook eval  --strategy ppo --weights ppo.pt --games 50
```

That command produced the bundled `ppo.pt`: 200,000 games in ~1 hour on a GTX
1070 (512 games in parallel, ~70k moves/s including learning), with the
learning rate decaying linearly to 0. During training it *samples* its moves
and ended around 43k points per game; played with its most likely move (what
`select_move` does) it is clearly stronger:

| On fresh games (300) | avg score | reaches 2048 | reaches 4096 | reaches 8192 |
|---|--:|--:|--:|--:|
| `ppo` (most likely move) | 55,946 | 92% | 60% | 7% |
| `dqn` (greedy, no search) | 41,229 | 79% | 48% | 0% |
| `dqn --depth 1` | 80,402 | 99% | 95% | 14% |

So, on this game, the policy beats the value network when neither searches.
But a policy only ranks moves, while `dqn`'s afterstate value plugs straight into
expectimax, and with one level of lookahead `dqn` is still the strongest player
in the catalog. (A fair caveat: PPO's learning rate decayed from the start, while
`dqn` only got one manual drop at the end; both could probably be pushed further.)
