"""Neural-network building blocks shared by the learning strategies.

Used by deep RL (:mod:`dqn <playbook.strategies.learning.reinforcement.deep.dqn>`,
a *value* network) and supervised imitation (:mod:`imitation
<playbook.strategies.learning.supervised.imitation>`, a *policy* network). This
module imports torch, so the agents import it lazily: the rest of the package
runs without it. Install with::

    pip install "playbook-2048[deep]"   # or: pip install torch
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_EXP = 16  # one-hot planes: empty, 2, 4, ..., 32768


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def one_hot_boards(boards, device, max_exp=MAX_EXP):
    """Encode ``(N, 4, 4)`` exponents as ``(N, max_exp, 4, 4)`` one-hot planes.

    One plane per tile value: a 1024 and a 2048 are different *kinds* of tile,
    not "twice as much" of something, so they get separate inputs.
    """
    b = torch.as_tensor(np.asarray(boards, dtype=np.int64), device=device)
    b = b.clamp_(0, max_exp - 1)
    return F.one_hot(b, max_exp).permute(0, 3, 1, 2).float()


class BoardNet(nn.Module):
    """One-hot board in, ``outputs`` numbers out (1 = a value, 4 = a policy).

    Two architectures, picked with ``arch``:

    * ``"mlp"``: the 256 one-hot inputs straight into fully connected layers.
      No built-in notion of rows and columns -- it has to learn them -- but it is
      a few big matrix products, which a GPU runs very fast.
    * ``"conv"``: tiles only ever interact with neighbours along a row or a
      column, so two conv layers look at horizontal (1x2) and vertical (2x1)
      pairs, and at pairs of pairs, before an MLP head -- the same inductive bias
      as the row/square tuples of the n-tuple network. Better per example, but
      convolutions over 4x4 maps use a GPU poorly (~3x slower on a GTX 1070).
    """

    def __init__(self, outputs=1, arch="mlp", channels=128, hidden=512, max_exp=MAX_EXP):
        super().__init__()
        self.config = dict(outputs=outputs, arch=arch, channels=channels, hidden=hidden,
                           max_exp=max_exp)
        if arch == "mlp":
            self.body = nn.Sequential(nn.Flatten(), nn.Linear(16 * max_exp, 2 * hidden), nn.ReLU())
            flat = 2 * hidden
        elif arch == "conv":
            self.h1 = nn.Conv2d(max_exp, channels, (1, 2))
            self.v1 = nn.Conv2d(max_exp, channels, (2, 1))
            self.h2 = nn.Conv2d(channels, channels, (1, 2))
            self.v2 = nn.Conv2d(channels, channels, (2, 1))
            # feature maps: h1 4x3, v1 3x4; then h2/v2 of each: 4x2, 3x3, 3x3, 2x4
            flat = channels * (12 + 12 + 8 + 9 + 9 + 8)
        else:
            raise ValueError(f"unknown arch {arch!r}; use 'mlp' or 'conv'")
        self.head = nn.Sequential(nn.Linear(flat, hidden), nn.ReLU(), nn.Linear(hidden, outputs))

    def _conv_features(self, x):
        h, v = F.relu(self.h1(x)), F.relu(self.v1(x))
        maps = [h, v, F.relu(self.h2(h)), F.relu(self.v2(h)),
                F.relu(self.h2(v)), F.relu(self.v2(v))]
        return torch.cat([m.flatten(1) for m in maps], dim=1)

    def forward(self, x):
        features = self.body(x) if self.config["arch"] == "mlp" else self._conv_features(x)
        return self.head(features)


def save_net(net, path, **extra):
    """Checkpoint = architecture config + weights (+ anything in ``extra``)."""
    torch.save({"config": net.config, "state_dict": net.state_dict(), **extra}, path)


def load_net(path, device):
    """Inverse of :func:`save_net`: returns ``(net, checkpoint_dict)``."""
    ckpt = torch.load(path, map_location=device)
    net = BoardNet(**ckpt["config"]).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt
