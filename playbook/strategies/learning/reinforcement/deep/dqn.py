"""Deep Q-Network for 2048, learned on afterstates.

The *deep* reinforcement-learning chapter: the value function is a neural net
(:class:`~playbook.strategies.learning.networks.BoardNet`) instead of lookup
tables. It keeps the ingredients that made DQN work on Atari:

  * **experience replay**: transitions go into a large buffer and the net learns
    from random minibatches instead of from the (highly correlated) last moves;
  * a **target network**: a frozen copy of the net computes the TD targets and
    is refreshed every ``target_every`` updates, so the net does not chase its
    own moving predictions;
  * **epsilon-greedy** exploration -- off by default (``epsilon=0``): the random
    spawns already make every game different, and in 2048 a random move is often
    a costly one.

And two changes that matter a lot for 2048:

**Afterstates.** A textbook DQN maps a board to four Q-values, so it must also
*learn* what each slide does. But a slide is deterministic and we can simulate
it, so we write

    Q(s, a) = r(s, a) + V(afterstate(s, a))

and the network only learns ``V``: how good the board is right after our move,
before the random tile appears (the n-tuple learner's idea, with a neural net in
place of the tables).

**n-step returns.** The 1-step target ``V(after_t) <- max_a' Q_target(s_t+1, a')``
moves information back one move per target refresh. A game lasts thousands of
moves, so for a long time the net only "sees" a few moves ahead and plays like a
greedy heuristic (it predicts ~20 points from an opening board that is worth
thousands). Summing the points actually scored over the next ``n`` moves before
bootstrapping fixes that:

    V(after_t)  <-  r_t+1 + ... + r_t+n-1  +  max_a' [ r(s_t+n, a') + V_target(afterstate(s_t+n, a')) ]

(no bootstrap term if the game ended first).

Training plays hundreds of games at once on :class:`~playbook.game.vector.VecSimEnv`
so the GPU always has a full batch. At play time ``depth`` adds expectimax
lookahead on top of the learned value (``depth=1``: average over every possible
spawn after each move), which trades speed for strength.
"""
import copy
import time

import numpy as np

from playbook.game.board import Move
from playbook.game.vector import VecSimEnv, random_symmetry
from playbook.strategies.base import Strategy, Trainable
from playbook.strategies.search.lookahead import expectimax_q

#: the net predicts value in thousands of points, keeping its outputs O(1-100)
VALUE_SCALE = 1000.0
#: boards per forward pass (bounds GPU memory during deep lookahead)
CHUNK = 65_536


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "DQNStrategy needs PyTorch. Install with: pip install 'playbook-2048[deep]'"
        ) from exc


class NStepWindow:
    """Turns the step-by-step stream of a :class:`VecSimEnv` into n-step transitions.

    Remembers the last ``n`` afterstates of every game together with the game's
    score right after each one, so "points scored since" is a subtraction. An
    afterstate leaves the window as ``(afterstate, points since, board, done)``:
    with ``done=False`` once it is ``n`` moves old, or ``done=True`` (and the
    points until the end) if its game finishes first.
    """

    def __init__(self, n, n_envs):
        self.n, self.t = n, 0
        self.after = np.zeros((n, n_envs, 4, 4), dtype=np.int8)
        self.score = np.zeros((n, n_envs), dtype=np.int64)
        self.live = np.zeros((n, n_envs), dtype=bool)

    def push(self, after, score_now, boards, dones):
        slot = self.t % self.n
        self.after[slot], self.score[slot], self.live[slot] = after, score_now, True
        self.t += 1
        out = []
        if dones.any():                       # finished games: flush everything pending
            j, i = np.nonzero(self.live & dones)
            out.append((self.after[j, i], score_now[i] - self.score[j, i], boards[i],
                        np.ones(len(i), dtype=bool)))
            self.live[:, dones] = False
        oldest = self.t % self.n              # the slot the next step overwrites
        i = np.flatnonzero(self.live[oldest])
        out.append((self.after[oldest, i], score_now[i] - self.score[oldest, i], boards[i],
                    np.zeros(len(i), dtype=bool)))
        self.live[oldest, i] = False
        return [np.concatenate(parts) for parts in zip(*out)]


class ReplayBuffer:
    """Ring buffer of ``(afterstate, points, next_board, done)`` n-step transitions."""

    def __init__(self, capacity):
        self.after = np.zeros((capacity, 4, 4), dtype=np.int8)
        self.points = np.zeros(capacity, dtype=np.float64)
        self.next = np.zeros((capacity, 4, 4), dtype=np.int8)
        self.done = np.zeros(capacity, dtype=bool)
        self.capacity, self.size, self.pos = capacity, 0, 0

    def add(self, after, points, nxt, done):
        idx = (self.pos + np.arange(len(after))) % self.capacity
        self.after[idx], self.points[idx], self.next[idx], self.done[idx] = after, points, nxt, done
        self.pos = (self.pos + len(after)) % self.capacity
        self.size = min(self.size + len(after), self.capacity)

    def sample(self, n, rng):
        idx = rng.integers(self.size, size=n)
        return self.after[idx], self.points[idx], self.next[idx], self.done[idx]

    def __len__(self):
        return self.size


class DQNStrategy(Strategy, Trainable):
    name = "dqn"

    def __init__(self, depth=0, net=None, device=None, seed=None, **net_config):
        _require_torch()
        import torch

        from playbook.strategies.learning.networks import BoardNet, default_device
        if seed is not None:
            torch.manual_seed(seed)
        self.device = device or default_device()
        self.net = (net or BoardNet(outputs=1, **net_config)).to(self.device).eval()
        self.depth = depth
        self.rng = np.random.default_rng(seed)

    # -- evaluating ----------------------------------------------------------
    def _values(self, afterstates, net):
        """``V`` of a batch of afterstates, in points."""
        import torch

        from playbook.strategies.learning.networks import one_hot_boards
        if len(afterstates) == 0:
            return np.zeros(0)
        out = []
        with torch.no_grad():
            for i in range(0, len(afterstates), CHUNK):
                x = one_hot_boards(afterstates[i:i + CHUNK], self.device)
                out.append(net(x).squeeze(1))
        return torch.cat(out).double().cpu().numpy() * VALUE_SCALE

    def _q(self, boards, net, depth=0):
        """``Q(s, a) = r + V(afterstate)`` in points, with ``depth`` levels of
        expectimax on top; ``-inf`` for illegal moves."""
        return expectimax_q(boards, lambda after: self._values(after, net), depth)

    def move_values(self, boards, depth=None):
        """``Q(s, a)`` for a batch of boards at this agent's lookahead ``depth``."""
        return self._q(np.asarray(boards), self.net, self.depth if depth is None else depth)

    # -- playing -------------------------------------------------------------
    def select_move(self, board, legal):
        q = self.move_values(board[None])[0]
        self.last_scores = {Move(m): float(q[m]) for m in legal}
        return max(sorted(legal), key=lambda m: q[m])

    def select_moves(self, boards):
        return self.move_values(boards).argmax(axis=1)

    # -- learning ------------------------------------------------------------
    def observe(self, transition):
        raise NotImplementedError(
            "DQN learns from minibatches of a replay buffer; use train(), which "
            "plays its own batched games")

    def _learn(self, buffer, target, optimizer, batch_size):
        import torch
        import torch.nn.functional as F

        from playbook.strategies.learning.networks import one_hot_boards
        after, points, nxt, done = buffer.sample(batch_size, self.rng)
        bootstrap = self._q(nxt, target).max(axis=1)
        y = points + np.where(done, 0.0, bootstrap)
        y = torch.as_tensor(y / VALUE_SCALE, dtype=torch.float32, device=self.device)
        # every rotation/reflection of an afterstate is worth the same
        x = one_hot_boards(random_symmetry(after, self.rng), self.device)
        loss = F.mse_loss(self.net(x).squeeze(1), y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        optimizer.step()
        return loss.item()

    def train(self, env=None, episodes=50_000, verbose=False, n_envs=256,
              batch_size=1024, lr=1e-4, n_step=10, buffer_size=1_000_000,
              learn_start=50_000, target_every=250, epsilon=0.0, explore_frac=0.1,
              report_every=1_000, checkpoint=None, seed=None):
        """Train for ``episodes`` finished games; return their scores.

        ``env`` is accepted for the :class:`Trainable` API but not used: a single
        Python env plays ~3k moves/s, far too few to keep a GPU busy, so training
        runs ``n_envs`` games at once on a :class:`VecSimEnv` (same rules). Each
        step plays one greedy move in every game (random with probability
        ``epsilon``, decayed linearly to 0 over the first ``explore_frac`` of the
        games), then takes one gradient step on a replay minibatch.

        Deep RL is noisy -- a network can get worse for a while -- so every
        ``report_every`` games the net is kept if its games averaged the best score
        so far (and saved to ``checkpoint``, if given). Training ends on that best
        net.
        """
        import torch

        vec = VecSimEnv(n_envs, seed=seed)
        window = NStepWindow(n_step, n_envs)
        buffer = ReplayBuffer(buffer_size)
        target = copy.deepcopy(self.net).eval()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.net.train()

        probe = VecSimEnv(256, seed=12345).boards   # fixed opening boards for the report
        scores, tiles, losses = [], [], []
        best_avg, best_state = -1.0, None
        updates, moves_played, start = 0, 0, time.time()
        while len(scores) < episodes:
            eps = epsilon * max(0.0, 1.0 - len(scores) / max(1.0, explore_frac * episodes))
            moves = self._q(vec.boards, self.net).argmax(axis=1)
            explore = self.rng.random(n_envs) < eps
            if explore.any():
                legal = vec.afterstates()[2][explore]
                keys = np.where(legal, self.rng.random(legal.shape), -1.0)
                moves[explore] = keys.argmax(axis=1)

            next_boards, _, dones, info = vec.step(moves)
            score_now = vec.scores.copy()
            score_now[dones] = info.get("final_score", [])
            buffer.add(*window.push(info["afterstate"], score_now, next_boards, dones))
            moves_played += n_envs

            if len(buffer) >= learn_start:
                losses.append(self._learn(buffer, target, optimizer, batch_size))
                updates += 1
                if updates % target_every == 0:
                    target.load_state_dict(self.net.state_dict())

            if dones.any():
                before = len(scores)
                scores.extend(info["final_score"].tolist())
                top = info["final_board"].reshape(-1, 16).max(axis=1).astype(np.int64)
                tiles.extend((1 << top).tolist())
                if len(scores) // report_every > before // report_every:
                    if verbose:
                        self._report(scores, tiles, losses, report_every, eps, moves_played,
                                     updates, start, probe)
                    losses.clear()
                    avg = float(np.mean(scores[-report_every:]))
                    if avg > best_avg:
                        best_avg = avg
                        best_state = copy.deepcopy(self.net.state_dict())
                        if checkpoint:
                            self.save(checkpoint)
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return scores

    def _report(self, scores, tiles, losses, window, eps, moves_played, updates, start, probe):
        s, t = np.array(scores[-window:]), np.array(tiles[-window:])
        elapsed = time.time() - start
        loss = f"{np.mean(losses):.3f}" if losses else "  -  "
        # what the net expects to score from an opening board, vs what it scores
        predicted = self._q(probe, self.net).max(axis=1).mean()
        print(f"  games {len(scores):>8}  avg {s.mean():>7.0f} (pred {predicted:>7.0f})  "
              f"2048 {100 * (t >= 2048).mean():>4.1f}%  4096 {100 * (t >= 4096).mean():>4.1f}%  "
              f"top {t.max():>5}  loss {loss}  eps {eps:.3f}  updates {updates:>7}  "
              f"{moves_played / elapsed:>6.0f} moves/s  {elapsed / 60:>5.1f} min", flush=True)

    # -- persistence ---------------------------------------------------------
    def save(self, path):
        from playbook.strategies.learning.networks import save_net
        save_net(self.net, path, value_scale=VALUE_SCALE)

    @classmethod
    def load(cls, path, device=None, **config):
        _require_torch()
        from playbook.strategies.learning.networks import default_device, load_net
        device = device or default_device()
        net, _ = load_net(path, device)
        return cls(net=net, device=device, **config)
