"""Proximal Policy Optimization (PPO): learn the policy itself.

:mod:`dqn` learns *values* and derives its policy from them (play the
best-valued move). Policy-gradient methods learn the **policy** directly: a
network outputs a probability for each move, and training raises the
probability of moves that turned out better than expected and lowers the rest.
PPO (Schulman et al., 2017) is the standard recipe, and the most widely used RL
algorithm today:

  * **actor-critic**: the same network also predicts ``V(s)``, the return
    expected from the board. A move's *advantage* is how much better than
    ``V(s)`` things actually went -- the signal the policy follows.
  * **GAE** (generalized advantage estimation, ``lam``): blends the 1-step
    estimate (low variance, biased by ``V``) with the full observed return
    (unbiased, noisy).
  * **the clipped objective**: an update may change a move's probability by at
    most ``clip`` (20%) relative to the policy that collected the data, so
    several epochs over the same batch do not wreck the policy.
  * illegal moves are masked out of the softmax, so they are never sampled.

Data comes from ``n_envs`` games played at once on
:class:`~playbook.game.vector.VecSimEnv`, ``n_steps`` moves each, then the batch
is used for a few epochs and thrown away (*on-policy*: unlike DQN's replay
buffer, old data was collected by a different policy and cannot be reused).

Value-based vs policy-based, on this game: PPO's policy, trained for an hour
with a decaying learning rate, plays better than ``dqn`` *without* search. But a
policy only says which move to prefer, while ``dqn``'s afterstate value plugs
straight into expectimax (``dqn --depth 1``), which is still the strongest player
in the catalog.
"""
import copy
import time

import numpy as np

from playbook.game.board import Move
from playbook.game.vector import VecSimEnv, expand
from playbook.strategies.base import Strategy, Trainable

#: returns are learned in thousands of points
VALUE_SCALE = 1000.0
#: logit given to illegal moves (probability exactly 0, no NaNs)
_MASKED = -1e9


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PPOStrategy needs PyTorch. Install with: pip install 'playbook-2048[deep]'"
        ) from exc


class PPOStrategy(Strategy, Trainable):
    name = "ppo"

    def __init__(self, net=None, device=None, seed=None, **net_config):
        _require_torch()
        import torch

        from playbook.strategies.learning.networks import BoardNet, default_device
        if seed is not None:
            torch.manual_seed(seed)
        self.device = device or default_device()
        # 4 policy logits (UDLR) + 1 value, on a shared body
        self.net = (net or BoardNet(outputs=5, **net_config)).to(self.device).eval()
        self.rng = np.random.default_rng(seed)

    # -- the network ---------------------------------------------------------
    def _forward(self, boards, legal):
        """Masked logits ``(N, 4)`` and values ``(N,)`` as torch tensors."""
        import torch

        from playbook.strategies.learning.networks import one_hot_boards
        out = self.net(one_hot_boards(boards, self.device))
        mask = torch.as_tensor(legal, device=self.device)
        return out[:, :4].masked_fill(~mask, _MASKED), out[:, 4]

    def _logits(self, boards):
        import torch
        _, _, legal = expand(boards)
        with torch.no_grad():
            logits, _ = self._forward(boards, legal)
        return logits.double().cpu().numpy()

    # -- playing -------------------------------------------------------------
    def select_moves(self, boards):
        return self._logits(boards).argmax(axis=1)   # play the most likely move

    def select_move(self, board, legal):
        logits = self._logits(board[None])[0]
        moves = sorted(legal)
        p = np.exp(logits[moves] - logits[moves].max())
        self.last_scores = {Move(m): 100 * float(v) for m, v in zip(moves, p / p.sum())}
        return max(moves, key=lambda m: logits[m])

    # -- learning ------------------------------------------------------------
    def observe(self, transition):
        raise NotImplementedError("PPO learns from batches of its own games; use train()")

    def _collect(self, vec, n_steps):
        """Play ``n_steps`` sampled moves in every game; return the batch (numpy)."""
        import torch
        n = vec.n
        boards = np.empty((n_steps, n, 4, 4), dtype=np.int8)
        legal = np.empty((n_steps, n, 4), dtype=bool)
        actions = np.empty((n_steps, n), dtype=np.int64)
        logps = np.empty((n_steps, n), dtype=np.float32)
        values = np.empty((n_steps + 1, n), dtype=np.float32)
        rewards = np.empty((n_steps, n), dtype=np.float32)
        dones = np.empty((n_steps, n), dtype=bool)
        finished = []
        with torch.no_grad():
            for t in range(n_steps):
                boards[t], legal[t] = vec.boards, vec.afterstates()[2]
                logits, value = self._forward(boards[t], legal[t])
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                actions[t] = action.cpu().numpy()
                logps[t] = dist.log_prob(action).cpu().numpy()
                values[t] = value.cpu().numpy()
                _, reward, done, info = vec.step(actions[t])
                rewards[t], dones[t] = reward / VALUE_SCALE, done
                if done.any():
                    tops = info["final_board"].reshape(-1, 16).max(axis=1).astype(np.int64)
                    finished += list(zip(info["final_score"].tolist(), (1 << tops).tolist()))
            _, value = self._forward(vec.boards, vec.afterstates()[2])
            values[n_steps] = value.cpu().numpy()
        return boards, legal, actions, logps, values, rewards, dones, finished

    @staticmethod
    def _advantages(values, rewards, dones, gamma, lam):
        """GAE: ``A_t = sum_k (gamma*lam)^k * delta_t+k`` with
        ``delta_t = r_t + gamma * V(s_t+1) - V(s_t)`` (no bootstrap past a game over)."""
        adv = np.zeros_like(rewards)
        running = np.zeros(rewards.shape[1], dtype=np.float32)
        for t in reversed(range(len(rewards))):
            alive = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * alive - values[t]
            running = delta + gamma * lam * alive * running
            adv[t] = running
        return adv, adv + values[:-1]

    def _update(self, batch, optimizer, epochs, minibatch, clip, value_coef, entropy_coef):
        import torch
        import torch.nn.functional as F
        boards, legal, actions, old_logp, adv, returns = batch
        stats = []
        for _ in range(epochs):
            for idx in np.array_split(self.rng.permutation(len(boards)),
                                      max(1, len(boards) // minibatch)):
                logits, value = self._forward(boards[idx], legal[idx])
                dist = torch.distributions.Categorical(logits=logits)
                a = torch.as_tensor(actions[idx], device=self.device)
                ratio = torch.exp(dist.log_prob(a) - torch.as_tensor(old_logp[idx], device=self.device))
                A = torch.as_tensor(adv[idx], device=self.device)
                A = (A - A.mean()) / (A.std() + 1e-8)
                policy_loss = -torch.min(ratio * A, ratio.clamp(1 - clip, 1 + clip) * A).mean()
                value_loss = F.mse_loss(value, torch.as_tensor(returns[idx], device=self.device))
                entropy = dist.entropy().mean()
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                optimizer.step()
                stats.append((value_loss.item(), entropy.item()))
        return np.mean(stats, axis=0)

    def train(self, env=None, episodes=100_000, verbose=False, n_envs=512, n_steps=128,
              lr=2.5e-4, gamma=0.995, lam=0.95, epochs=4, minibatch=8192, clip=0.2,
              value_coef=0.5, entropy_coef=0.01, report_every=1_000, checkpoint=None,
              seed=None):
        """Train for ``episodes`` finished games; return their scores.

        Each iteration plays ``n_steps`` moves in ``n_envs`` games (sampling from the
        policy), computes GAE advantages, and runs ``epochs`` passes of clipped PPO
        updates over that batch. The learning rate decays linearly from ``lr`` to 0
        over the ``episodes`` games. As with :class:`DQNStrategy`, the network whose
        games averaged the best score is kept (and saved to ``checkpoint``). ``env``
        is unused: the games run on a batched simulator.
        """
        import torch
        vec = VecSimEnv(n_envs, seed=seed)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.net.train()
        scores, tiles = [], []
        best_avg, best_state = -1.0, None
        moves_played, start = 0, time.time()
        while len(scores) < episodes:
            boards, legal, actions, logps, values, rewards, dones, finished = \
                self._collect(vec, n_steps)
            moves_played += n_steps * n_envs
            for group in optimizer.param_groups:
                group["lr"] = lr * max(0.0, 1.0 - len(scores) / episodes)
            adv, returns = self._advantages(values, rewards, dones, gamma, lam)
            flat = lambda x: x.reshape(n_steps * n_envs, *x.shape[2:])  # noqa: E731
            value_loss, entropy = self._update(
                tuple(map(flat, (boards, legal, actions, logps, adv, returns))),
                optimizer, epochs, minibatch, clip, value_coef, entropy_coef)

            before = len(scores)
            for score, tile in finished:
                scores.append(score)
                tiles.append(tile)
            if len(scores) // report_every > before // report_every:
                avg = float(np.mean(scores[-report_every:]))
                if verbose:
                    t = np.array(tiles[-report_every:])
                    elapsed = time.time() - start
                    print(f"  games {len(scores):>8}  avg {avg:>7.0f}  "
                          f"2048 {100 * (t >= 2048).mean():>4.1f}%  "
                          f"4096 {100 * (t >= 4096).mean():>4.1f}%  top {t.max():>5}  "
                          f"entropy {entropy:.3f}  value loss {value_loss:.4f}  "
                          f"{moves_played / elapsed:>6.0f} moves/s  {elapsed / 60:>5.1f} min",
                          flush=True)
                if avg > best_avg:
                    best_avg, best_state = avg, copy.deepcopy(self.net.state_dict())
                    if checkpoint:
                        self.save(checkpoint)
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return scores

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
