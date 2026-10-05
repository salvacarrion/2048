# Tabular reinforcement learning

The textbook algorithm, kept here for what it teaches: **where tables break**.

| File | Idea | Status |
|---|---|---|
| `q_learning.py` | Q-learning on a table indexed by a coarse summary of the board | implemented + trained table (`qlearning.pkl`) |

```text
Q(s, a) += alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))
```

2048 has far more boards than any table can hold, and a table never
generalizes: what it learns about one board says nothing about the board next to
it. So the table is indexed by an **abstraction** of the board, and you can try
both ends of the trade-off:

```bash
python -m playbook train --strategy qlearning --episodes 20000 --save q.pkl   # "coarse" (default)
python -m playbook eval  --strategy qlearning --weights q.pkl --games 100
```

- **`raw`** (the exact board): after 2,000 games the table holds ~640k entries,
  almost every board it meets is new, and it plays *worse than random* (~790
  points vs ~1,080).
- **`coarse`** (default): the kind of cell the biggest tile sits in, how many
  cells are free, and what each move would do (illegal / slide / small or big
  merge / drags the biggest tile away). About 13k possible keys, so the table
  fills up and learns: roughly twice the score of random play, but still below
  the one-line `greedy` player, because boards that need different moves share
  a key.

Two details that mattered: the step size defaults to `1 / N(s, a)` (each entry is
the average of its targets; with so many boards behind one key a fixed `alpha`
never settles), and a short horizon (`gamma=0.5`) beats a long one, since the
noise of `max Q(s')` grows with every step it looks ahead.

The fix is to generalize across boards: many small overlapping tables
([`ntuple/`](../ntuple/)) or a neural network ([`deep/`](../deep/)).
