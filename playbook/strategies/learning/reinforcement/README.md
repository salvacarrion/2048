# Reinforcement learning

Learn a value (or policy) from the reward of self-play. The key 2048-specific
idea is to learn on **afterstates** — the board *after* your slide/merge but
*before* the random tile appears. The environment hands you that board in
`info["afterstate"]`, and value-based RL there is both simpler and stronger than
learning on full states.

Organized by how the value function is *represented*:

| Sub-family | Representation | Status |
|---|---|---|
| [`tabular/`](tabular/) | one entry per (abstracted) state | implemented — shows *why* tables don't scale |
| [`ntuple/`](ntuple/) | lookup tables over cell groups | **fully worked** — learns strong play on CPU, no torch |
| [`deep/`](deep/) | neural network: DQN (values of afterstates) and PPO (the policy itself) | implemented — trains on a GPU |

Read them in that order: `tabular/` is the textbook algorithm and shows where it
breaks (a table cannot generalize, so it either aliases very different boards or
never sees the same board twice); `ntuple/` is the canonical, lightweight fix for
this game (Szubert & Jaśkowski, 2014) and the best bang for the buck; `deep/`
replaces the tables with a neural network, in both flavours of deep RL:
value-based (`dqn`) and policy-based (`ppo`).

`ntuple` and `dqn` both learn `V(afterstate)`, so both accept `--depth` to search
a little before trusting it ([`search/lookahead.py`](../../search/lookahead.py)):
`--depth 1` averages over every possible spawn after each move.
