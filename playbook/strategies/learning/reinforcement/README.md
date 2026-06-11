# Reinforcement learning

Learn a value (or policy) from the reward of self-play. The key 2048-specific
idea is to learn on **afterstates** — the board *after* your slide/merge but
*before* the random tile appears. The environment hands you that board in
`info["afterstate"]`, and value-based RL there is both simpler and stronger than
learning on full states.

Organized by how the value function is *represented*:

| Sub-family | Representation | Status |
|---|---|---|
| [`ntuple/`](ntuple/) | lookup tables over cell groups | **fully worked** — learns strong play on CPU, no torch |
| [`tabular/`](tabular/) | one entry per state | scaffold (motivates why tables don't scale) |
| [`deep/`](deep/) | neural network (DQN, policy gradient) | scaffold + network + lazy torch |

Start with `ntuple/` — it is the canonical, lightweight RL approach for this
game (Szubert & Jaśkowski, 2014) and the best bang for the buck.
