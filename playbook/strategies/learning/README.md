# Learning strategies

Players that **improve from data or experience** rather than from a hand-written
heuristic. Two sub-families:

- [`supervised/`](supervised/) — learn from labelled examples. The example here
  is **imitation**: log a strong teacher's moves and train a model to copy them
  (distilling slow search into a fast reactive player).
- [`reinforcement/`](reinforcement/) — learn from the reward of self-play. Split
  by representation: `tabular/`, `ntuple/` (the classic 2048 method, fully
  worked), and `deep/` (neural networks).

All of them mix in
[`Trainable`](../base.py): a `train()` loop plus `save()`/`load()`. Train from the
CLI:

```bash
python -m playbook train --strategy ntuple --episodes 20000 --save ntuple.npz
python -m playbook eval  --strategy ntuple --weights ntuple.npz --games 50
```
