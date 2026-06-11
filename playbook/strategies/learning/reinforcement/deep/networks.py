"""Neural-network building blocks for the deep-RL strategies.

Kept next to the agents that use them so the "deep RL" chapter is self-contained.
torch is imported lazily (only when this module is used), so the rest of the
package runs without it. Install with::

    pip install "playbook[deep]"   # or: pip install torch
"""


def one_hot_board(board, max_exp=16):
    """Encode a 4x4 board of exponents as a (max_exp, 4, 4) one-hot tensor."""
    import torch
    b = torch.as_tensor(board, dtype=torch.long).clamp(0, max_exp - 1)
    planes = torch.zeros(max_exp, 4, 4)
    planes.scatter_(0, b.unsqueeze(0), 1.0)
    return planes


def build_qnetwork(max_exp=16, hidden=256):
    """A small conv+MLP that maps a one-hot board to 4 move-values."""
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(max_exp, 128, kernel_size=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.LazyLinear(hidden),
        nn.ReLU(),
        nn.Linear(hidden, 4),  # one Q-value per move (UDLR)
    )
