"""Imitation learning: distill a strong, slow teacher into a fast reactive player.

Supervised learning instead of reward: collect ``(board -> move)`` examples from
a strong teacher, then train a classifier (a policy network) to predict the
teacher's move. The student plays with a single forward pass -- no simulation,
no lookahead -- so this is a cheap way to "distill" a slow search player into a
fast one.

Three details make the difference between copying and *learning* the teacher:

  * **DAgger** (Ross et al., 2011). A student trained only on the teacher's games
    never sees its own mistakes: one bad move takes it to boards the teacher
    would never have reached, where it has no data. So after the first round the
    *student* plays and the teacher only labels the boards it visits; every round
    is added to the dataset and the student is retrained on all of it.
  * **Soft targets** (distillation, Hinton et al., 2015). "The teacher played
    LEFT" does not say whether RIGHT was nearly as good or a disaster, and a
    student that agrees with the teacher ~60% of the time spends most of its
    disagreements on near-ties -- the rare blunders are what end games. If the
    teacher can score every move (``move_values``, as ``dqn`` and ``ntuple`` do),
    the student learns the whole distribution ``softmax(Q / temperature)``
    instead: moves a few points apart share the probability, a move hundreds of
    points worse gets none.
  * **Symmetry**. Every example is shown under a random rotation/reflection, with
    the moves mapped accordingly: 8x the data for free.

Any :class:`Strategy` can be the teacher. Data is collected on a batched
simulator through :meth:`Strategy.select_moves`, so a neural teacher (e.g.
``dqn`` with lookahead) labels a whole batch of boards per forward pass.
"""
import time

import numpy as np

from playbook.game.board import Move
from playbook.game.vector import VecSimEnv, expand, random_symmetry
from playbook.strategies.base import Strategy, Trainable


def teacher_targets(teacher, boards, temperature=50.0):
    """What the student should learn on each board: ``(moves, targets)``.

    ``targets`` is a ``(N, 4)`` distribution over moves: ``softmax(Q / temperature)``
    (``Q`` in points) when the teacher can score moves, else one-hot on its move.
    """
    if hasattr(teacher, "move_values"):
        q = teacher.move_values(boards)
        z = np.exp((q - q.max(axis=1, keepdims=True)) / temperature)   # illegal -> 0
        return q.argmax(axis=1), (z / z.sum(axis=1, keepdims=True)).astype(np.float32)
    moves = np.asarray(teacher.select_moves(boards))
    return moves, np.eye(4, dtype=np.float32)[moves]


def collect_demonstrations(teacher, games, player=None, n_envs=256, seed=None,
                           temperature=50.0):
    """Play ``games`` complete games and label every board with the teacher's choice.

    ``player`` makes the moves actually played (default: the teacher itself);
    the teacher only labels. Returns ``(boards, targets, scores)``: ``(M, 4, 4)``
    boards, their ``(M, 4)`` :func:`teacher_targets`, and the scores of the games.

    Every game is played to the end: stopping early would cut exactly the longest
    games, whose late, big-tile boards are the most valuable examples.
    """
    vec = VecSimEnv(min(n_envs, games), seed=seed)
    started = vec.n
    counted = np.ones(vec.n, dtype=bool)   # is this slot playing one of our games?
    boards, labels, scores = [], [], []
    while counted.any():
        idx = np.flatnonzero(counted)
        current = vec.boards[idx]
        teacher_moves, targets = teacher_targets(teacher, current, temperature)
        boards.append(current)
        labels.append(targets)
        moves = vec.afterstates()[2].argmax(axis=1)      # idle slots: any legal move
        moves[idx] = teacher_moves if player is None else player.select_moves(current)
        _, _, dones, info = vec.step(moves)
        for k, i in enumerate(np.flatnonzero(dones)):
            if counted[i]:
                scores.append(int(info["final_score"][k]))
                if started < games:
                    started += 1                         # this slot starts another game
                else:
                    counted[i] = False
    return np.concatenate(boards), np.concatenate(labels), scores


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ImitationStrategy needs PyTorch. Install with: pip install 'playbook-2048[deep]'"
        ) from exc


class ImitationStrategy(Strategy, Trainable):
    name = "imitation"

    def __init__(self, net=None, device=None, seed=None, **net_config):
        _require_torch()
        import torch

        from playbook.strategies.learning.networks import BoardNet, default_device
        if seed is not None:
            torch.manual_seed(seed)
        self.device = device or default_device()
        # conv by default: per example it imitates clearly better than the MLP, and
        # here data (the teacher's time), not GPU time, is the scarce resource
        net_config = {"arch": "conv", **net_config}
        self.net = (net or BoardNet(outputs=4, **net_config)).to(self.device).eval()
        self.rng = np.random.default_rng(seed)

    # -- playing -------------------------------------------------------------
    def _logits(self, boards):
        import torch

        from playbook.strategies.learning.networks import one_hot_boards
        with torch.no_grad():
            return self.net(one_hot_boards(boards, self.device)).double().cpu().numpy()

    def select_moves(self, boards):
        _, _, legal = expand(boards)
        return np.where(legal, self._logits(boards), -np.inf).argmax(axis=1)

    def select_move(self, board, legal):
        logits = self._logits(board[None])[0]
        moves = sorted(legal)
        p = np.exp(logits[moves] - logits[moves].max())
        # explain as the student's confidence in each move (percent)
        self.last_scores = {Move(m): 100 * float(v) for m, v in zip(moves, p / p.sum())}
        return max(moves, key=lambda m: logits[m])

    # -- learning ------------------------------------------------------------
    def observe(self, transition):
        raise NotImplementedError("Imitation learns from a teacher's dataset; use train()")

    def _fit(self, boards, targets, lrs, batch_size, optimizer):
        """One epoch of cross-entropy against the teacher's ``targets`` per
        learning rate in ``lrs``; returns how often the student's top move matched
        the teacher's in the last epoch."""
        import torch
        import torch.nn.functional as F

        from playbook.strategies.learning.networks import one_hot_boards
        self.net.train()
        for lr in lrs:
            for group in optimizer.param_groups:
                group["lr"] = lr
            order = self.rng.permutation(len(boards))
            correct = 0
            for i in range(0, len(order), batch_size):
                idx = order[i:i + batch_size]
                b, t = random_symmetry(boards[idx], self.rng, values=targets[idx])
                logits = self.net(one_hot_boards(b, self.device))
                target = torch.as_tensor(t, device=self.device)
                loss = F.cross_entropy(logits, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                correct += int((logits.argmax(dim=1) == target.argmax(dim=1)).sum())
        self.net.eval()
        return correct / len(boards)

    def train(self, env=None, episodes=3_000, teacher=None, rounds=3, epochs=4,
              batch_size=1024, lr=1e-3, temperature=50.0, n_envs=512, verbose=False,
              checkpoint=None, seed=None):
        """Distill ``teacher`` with DAgger; ``episodes`` games split over ``rounds``.

        Round 0: the teacher plays. Later rounds: the student plays, the teacher
        labels. After each round the student trains ``epochs`` passes over all the
        data so far, with a learning rate that decays from ``lr`` towards 0 (cosine)
        across all rounds. ``temperature`` (in points) sets how soft the targets
        are when the teacher can score moves. ``env`` is unused (data comes from a
        batched simulator). Returns the boards and targets of the whole dataset.
        """
        import torch
        if teacher is None:
            raise ValueError("ImitationStrategy.train needs a teacher strategy "
                             "(CLI: --teacher dqn --teacher-weights dqn.pt)")

        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        total = rounds * epochs
        schedule = lr * 0.5 * (1 + np.cos(np.pi * np.arange(total) / total))
        boards, targets = [], []
        per_round = max(1, episodes // rounds)
        start = time.time()
        for r in range(rounds):
            player = None if r == 0 else self
            b, t, scores = collect_demonstrations(teacher, per_round, player=player,
                                                  n_envs=n_envs, temperature=temperature,
                                                  seed=None if seed is None else seed + r)
            boards.append(b)
            targets.append(t)
            all_b, all_t = np.concatenate(boards), np.concatenate(targets)
            acc = self._fit(all_b, all_t, schedule[r * epochs:(r + 1) * epochs], batch_size,
                            optimizer)
            if verbose:
                who = "teacher" if player is None else "student"
                print(f"  round {r + 1}/{rounds}  {who} played {len(scores)} games "
                      f"(avg {np.mean(scores):.0f})  examples {len(all_b):,}  "
                      f"train acc {100 * acc:.1f}%  {(time.time() - start) / 60:.1f} min",
                      flush=True)
            if checkpoint:
                self.save(checkpoint)
        return all_b, all_t

    # -- persistence ---------------------------------------------------------
    def save(self, path):
        from playbook.strategies.learning.networks import save_net
        save_net(self.net, path)

    @classmethod
    def load(cls, path, device=None, **config):
        _require_torch()
        from playbook.strategies.learning.networks import default_device, load_net
        device = device or default_device()
        net, _ = load_net(path, device)
        return cls(net=net, device=device, **config)
