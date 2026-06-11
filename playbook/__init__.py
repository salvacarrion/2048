"""playbook — a didactic catalog of strategies for playing 2048.

Layout:
    game/        the rules + simulator + environments (the "world")
    heuristics/  reusable board-evaluation functions (the "ideas")
    strategies/  the players, grouped by technique:
        baselines/      random, greedy, manual
        search/         maximization, minimax, expectimax, mcts
        optimization/   metaheuristics that tune a parametric player
        learning/       supervised + reinforcement (incl. deep)
    evaluation/  run games, aggregate metrics, compare strategies
    registry.py  name -> strategy, so you can pick one without reading the code

The whole package speaks one tiny interface, ``Strategy.select_move(board,
legal) -> Move``, so trying a new idea means copying the closest file and
rewriting that one method.
"""
__version__ = "0.1.0"
