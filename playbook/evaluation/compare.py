"""Compare several strategies under identical conditions."""
from ..game.env import SimEnv
from ..registry import make_strategy
from .evaluate import evaluate
from .report import compare_table


def compare(names, env_factory=SimEnv, games=10, max_moves=10_000, configs=None):
    """Evaluate each named strategy and return a list of :class:`EvalReport`.

    ``configs`` optionally maps a strategy name to its constructor kwargs.
    """
    configs = configs or {}
    reports = []
    for name in names:
        strategy = make_strategy(name, **configs.get(name, {}))
        reports.append(evaluate(strategy, env_factory, games=games,
                                max_moves=max_moves, name=name))
    return reports


def compare_and_print(names, **kwargs):
    reports = compare(names, **kwargs)
    print(compare_table(reports))
    return reports
