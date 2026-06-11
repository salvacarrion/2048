"""Weighted combinations of feature heuristics.

Replaces the old module-level ``H_SCORES`` global (which was populated as an
import side effect). A ``CombinedHeuristic`` is an explicit, self-contained
object: you pass it the (feature, weight) pairs, it normalizes them, and it is
callable like any other heuristic.
"""
from . import features


class CombinedHeuristic:
    """A callable weighted sum of feature heuristics.

    >>> h = CombinedHeuristic([(features.gradient, 2.0), (features.max_free, 1.0)])
    >>> h(board)  # doctest: +SKIP
    """

    def __init__(self, terms, normalize=True):
        self.terms = list(terms)
        if normalize:
            total = sum(abs(w) for _, w in self.terms) or 1.0
            self.terms = [(f, w / total) for f, w in self.terms]

    def __call__(self, board):
        return sum(w * f(board) for f, w in self.terms)

    def __repr__(self):
        names = ", ".join(f"{f.__name__}={w:.3f}" for f, w in self.terms)
        return f"CombinedHeuristic({names})"


def default_heuristic():
    """The default weighted heuristic (the old tuned ``h_combined``)."""
    return CombinedHeuristic([
        (features.total_sum, 260),
        (features.gradient, 7989),
        (features.max_free, 6117),
    ])


# Named heuristics for the CLI / registry. Each is a zero-arg factory.
HEURISTICS = {
    "combined": default_heuristic,
    "approx_score": lambda: features.approx_score,
    "gradient": lambda: features.gradient,
    "gradient_corners": lambda: features.gradient_corners,
    "max_free": lambda: features.max_free,
}


def get_heuristic(name):
    """Resolve a heuristic by name into a callable ``board -> float``."""
    if name not in HEURISTICS:
        raise KeyError(f"unknown heuristic {name!r}; choices: {sorted(HEURISTICS)}")
    return HEURISTICS[name]()
