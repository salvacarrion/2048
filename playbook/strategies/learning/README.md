# Learning strategies

Players that **improve from data or experience** rather than from a hand-written
heuristic. Two sub-families:

- [`supervised/`](supervised/) — learn from labelled examples. The example here
  is **imitation**: log a strong teacher's moves and train a policy network to
  copy them (distilling slow search into a fast reactive player), with DAgger.
- [`reinforcement/`](reinforcement/) — learn from the reward of self-play. Split
  by representation: `tabular/` (Q-learning), `ntuple/` (the classic 2048
  method, fully worked), and `deep/` (DQN).

The neural strategies (`dqn`, `imitation`) share [`networks.py`](networks.py)
and train on the batched simulator ([`game/vector.py`](../../game/vector.py)),
which plays hundreds of games at once so the GPU always has a full batch.

All of them mix in
[`Trainable`](../base.py): a `train()` loop plus `save()`/`load()`. Train from the
CLI:

```bash
python -m playbook train --strategy ntuple --episodes 20000 --save ntuple.npz
python -m playbook eval  --strategy ntuple --weights ntuple.npz --games 50
python -m playbook train --strategy dqn --episodes 85000 --save dqn.pt    # GPU, ~2 h
python -m playbook train --strategy imitation --episodes 3000 --save imitation.pt \
    --teacher dqn --teacher-weights dqn.pt --teacher-depth 1             # GPU, ~30 min
```

The exact recipes behind the bundled weights, and what each one scores, are in
[`deep/README.md`](reinforcement/deep/README.md),
[`supervised/README.md`](supervised/README.md) and
[`tabular/README.md`](reinforcement/tabular/README.md).
