"""Deep Q-Network for 2048 — scaffold.

This is the *deep* reinforcement-learning chapter: the value function is a neural
net (see :mod:`networks`) instead of lookup tables. The interface is identical to
every other strategy, so once trained it slots straight into ``eval``/``compare``.

The wiring (interface, lazy torch import, encoder, greedy action selection over
legal moves) is provided. The training loop is left as the exercise — fill in
:meth:`DQNStrategy.train` with experience replay + a target network. The
afterstate signal in ``info["afterstate"]`` and ``move_reward`` make a value- or
Q-learning target straightforward.
"""
import random

from playbook.strategies.base import Strategy, Trainable


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "DQNStrategy needs PyTorch. Install with: pip install 'playbook[deep]'"
        ) from exc


class DQNStrategy(Strategy, Trainable):
    name = "dqn"

    def __init__(self, max_exp=16, seed=None):
        _require_torch()
        from .networks import build_qnetwork
        self.max_exp = max_exp
        self.model = build_qnetwork(max_exp=max_exp)
        self.model.eval()
        self.rng = random.Random(seed)

    def _q_values(self, board):
        import torch
        from .networks import one_hot_board
        with torch.no_grad():
            x = one_hot_board(board, self.max_exp).unsqueeze(0)
            return self.model(x).squeeze(0)

    def select_move(self, board, legal):
        q = self._q_values(board)
        # pick the highest-Q move that is legal
        ranked = sorted(range(4), key=lambda m: float(q[m]), reverse=True)
        for move in ranked:
            if move in legal:
                return move
        return self.rng.choice(sorted(legal))

    def observe(self, transition):
        raise NotImplementedError("TODO: push transition into the replay buffer")

    def train(self, env, episodes=10_000, verbose=False, **kwargs):
        raise NotImplementedError(
            "TODO: implement DQN training (epsilon-greedy rollouts on SimEnv, "
            "replay buffer, target network, Huber loss on r + gamma*max_a' Q_target)."
        )

    def save(self, path):
        import torch
        torch.save(self.model.state_dict(), path)

    @classmethod
    def load(cls, path):
        import torch
        agent = cls()
        agent.model.load_state_dict(torch.load(path))
        agent.model.eval()
        return agent
