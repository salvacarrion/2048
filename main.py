"""Backwards-compatible entry point.

The project is now the ``playbook`` package; this shim just forwards to its CLI.
Prefer running the CLI directly:

    python -m playbook list
    python -m playbook compare --strategies random,greedy,expectimax --games 10
    python -m playbook eval    --strategy expectimax --depth 3 --games 10
    python -m playbook play     --strategy mcts --env browser
    python -m playbook train    --strategy ntuple --episodes 20000 --save ntuple.npz
"""
from playbook.cli import main

if __name__ == "__main__":
    main()
