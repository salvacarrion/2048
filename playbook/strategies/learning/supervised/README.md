# Supervised learning

Learn from labelled examples instead of reward.

| File | Idea | Status |
|---|---|---|
| `imitation.py` | distill a strong, slow teacher into a fast policy network, with DAgger | implemented + trained student (`imitation.pt`) |

**Imitation** turns playing into classification: record `(board, move)` pairs
from a teacher and train a network to predict the teacher's move. The student
then plays with one forward pass per move: no simulation, no lookahead.

```bash
python -m playbook train --strategy imitation --episodes 3000 --save imitation.pt \
    --teacher dqn --teacher-weights playbook/strategies/learning/reinforcement/deep/dqn.pt --teacher-depth 1
python -m playbook eval  --strategy imitation --weights imitation.pt --games 50
```

That is exactly how the bundled `imitation.pt` was trained (~30 min on a GTX
1070): 3 rounds of 1,000 games (the teacher plays the first, the student the
other two), 5.7M labelled boards, a conv student trained 4 epochs per round with
a cosine-decayed learning rate.

| Player on fresh games (300) | avg score | reaches 2048 | reaches 4096 |
|---|--:|--:|--:|
| teacher: `dqn --depth 1` | 80,402 | 99% | 95% |
| student, soft targets (bundled) | 39,327 | 83% | 29% |
| student, hard labels (same recipe) | 28,635 | 67% | 9% |

The student plays with one forward pass per move and no search at all; for
comparison, the same `dqn` without its lookahead scores 41,229 (4096 in 48%).

Any registered strategy can be the teacher (`--teacher expectimax --teacher-depth 2`,
`--teacher ntuple --teacher-weights ... --teacher-depth 1`, ...). Demonstrations
are collected on the batched simulator through `Strategy.select_moves`, so a
neural teacher labels a whole batch of boards per forward pass.

What matters, in order:

1. **The teacher.** A student rarely beats its teacher, so distill the strongest
   player you have. `dqn` with one level of lookahead reaches 4096 in most games.
2. **Soft targets** (distillation, Hinton et al., 2015). The student agrees
   with the teacher on only ~60% of moves, and the median cost of a
   disagreement (judged by the teacher) is 0 points: most are near-ties. But 1%
   of its moves cost 750+ points, and those blunders end games. A teacher that
   can score every move (`move_values`: `dqn`, `ntuple`) is imitated through
   `softmax(Q / temperature)` instead of a one-hot label, so a move a few points
   worse keeps some probability and a blunder gets none. Same data, same
   network: 4096 in 29% of games instead of 9%.
3. **DAgger** (Ross et al., 2011). A student trained only on the teacher's games
   never sees its own mistakes: one bad move leads to boards the teacher never
   reached, where it has no data. So after the first round the *student* plays
   and the teacher labels the boards it visits; each round is added to the data
   and the student retrained on all of it.
4. **Symmetry.** Each example is shown under a random rotation/reflection, with
   the move remapped (a horizontal flip swaps LEFT and RIGHT): 8x the data.
5. **The network.** Small accuracy gains compound over thousands of moves: on
   the same 650k examples (from an `ntuple --depth 1` teacher) a conv student
   (66% agreement with the teacher) scored ~11.7k on average where an MLP (63%)
   scored ~6.8k. That is why the student is a conv net while `dqn` uses the
   faster MLP: here the teacher's time, not GPU time, is the scarce resource.
