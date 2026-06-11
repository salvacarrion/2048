"""Strategy registry: map a name to a factory so you can pick a player without
importing its class.

Each entry is ``name -> factory(**config) -> Strategy``. The CLI and the
``compare`` evaluator build players through here. To expose a new strategy, add
one line in :func:`_build_registry` (imports are local so optional dependencies
like torch are only loaded if that strategy is actually requested).
"""
from .heuristics.combined import get_heuristic


def _search(cls, **config):
    """Build a search strategy, resolving the heuristic name to a callable."""
    heuristic = config.pop("heuristic", "combined")
    if isinstance(heuristic, str):
        heuristic = get_heuristic(heuristic)
    return cls(heuristic=heuristic, **config)


def _build_registry():
    from .strategies.baselines import GreedyStrategy, ManualStrategy, RandomStrategy
    from .strategies.search import (
        ExpectimaxStrategy,
        MaximizationStrategy,
        MctsStrategy,
        MinimaxStrategy,
    )

    reg = {
        # baselines
        "random": lambda **c: RandomStrategy(**c),
        "greedy": lambda **c: GreedyStrategy(**c),
        "manual": lambda **c: ManualStrategy(),
        # search
        "maximization": lambda **c: _search(MaximizationStrategy, **c),
        "minimax": lambda **c: _search(MinimaxStrategy, **c),
        "expectimax": lambda **c: _search(ExpectimaxStrategy, **c),
        "mcts": lambda **c: _search(MctsStrategy, **c),
    }

    # Optimization & learning families: imported lazily so a missing optional
    # dependency (e.g. torch for DQN) only errors when that name is requested.
    def _ntuple(**c):
        from .strategies.learning.reinforcement.ntuple import NTupleStrategy
        weights = c.pop("weights", None)
        agent = NTupleStrategy.load(weights) if weights else NTupleStrategy(**c)
        return agent

    def _genetic(**c):
        from .strategies.optimization.genetic import GeneticStrategy
        return GeneticStrategy(**c)

    def _dqn(**c):
        from .strategies.learning.reinforcement.deep.dqn import DQNStrategy
        weights = c.pop("weights", None)
        return DQNStrategy.load(weights) if weights else DQNStrategy(**c)

    def _qlearning(**c):
        from .strategies.learning.reinforcement.tabular.q_learning import QLearningStrategy
        return QLearningStrategy(**c)

    def _imitation(**c):
        from .strategies.learning.supervised.imitation import ImitationStrategy
        return ImitationStrategy()

    reg["ntuple"] = _ntuple
    reg["genetic"] = _genetic
    reg["dqn"] = _dqn
    reg["qlearning"] = _qlearning
    reg["imitation"] = _imitation
    return reg


_REGISTRY = None


def _registry():
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY


def available():
    """Sorted list of registered strategy names."""
    return sorted(_registry())


def make_strategy(name, **config):
    """Instantiate a strategy by name with the given config."""
    reg = _registry()
    if name not in reg:
        raise KeyError(f"unknown strategy {name!r}; choices: {available()}")
    return reg[name](**config)
