# Deep reinforcement learning

The neural-network chapter. Same `Strategy` interface as everything else, but the
value/policy function is a net (see [`networks.py`](networks.py)) instead of
tables. torch is imported **lazily**, so the rest of the package runs without it:

```bash
pip install "playbook[deep]"     # installs torch
```

| File | Idea | Status |
|---|---|---|
| `dqn.py` | Deep Q-Network (value-based) | scaffold: interface + encoder + greedy action wired; training loop is the exercise |
| `networks.py` | one-hot encoder + conv/MLP Q-net | provided |

`policy_gradient.py` (REINFORCE / actor-critic) is a natural next file to add.
The afterstate signal and `move_reward` make value targets straightforward.
