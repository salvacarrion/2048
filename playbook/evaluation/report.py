"""Structured results for evaluation runs (dataclasses, not prints)."""
from collections import Counter
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GameResult:
    score: int
    max_tile: int          # displayed value, e.g. 2048
    moves: int
    elapsed: float


@dataclass(frozen=True)
class EvalReport:
    strategy: str
    games: list = field(default_factory=list)   # list[GameResult]
    elapsed: float = 0.0

    @property
    def avg_score(self):
        return sum(g.score for g in self.games) / len(self.games)

    @property
    def avg_moves(self):
        return sum(g.moves for g in self.games) / len(self.games)

    @property
    def best_score(self):
        return max(g.score for g in self.games)

    @property
    def moves_per_sec(self):
        total_moves = sum(g.moves for g in self.games)
        return total_moves / self.elapsed if self.elapsed else 0.0

    @property
    def tile_distribution(self):
        """Map of max tile reached -> count, highest first."""
        counts = Counter(g.max_tile for g in self.games)
        return dict(sorted(counts.items(), key=lambda kv: kv[0], reverse=True))

    def reach_rate(self, target=2048):
        """Fraction of games whose top tile was at least ``target`` (e.g. 2048)."""
        if not self.games:
            return 0.0
        return sum(g.max_tile >= target for g in self.games) / len(self.games)

    def summary(self):
        n = len(self.games)
        lines = [
            f"Strategy: {self.strategy}",
            f"  games:        {n}",
            f"  avg score:    {self.avg_score:.0f}",
            f"  best score:   {self.best_score}",
            f"  avg moves:    {self.avg_moves:.0f}",
            f"  speed:        {self.moves_per_sec:.1f} moves/s",
            "  max tile reached:",
        ]
        for tile, count in self.tile_distribution.items():
            lines.append(f"    {tile:>6}: {count:>3} ({100 * count / n:.0f}%)")
        return "\n".join(lines)


def compare_table(reports):
    """Render a list of EvalReport as a comparison table."""
    header = f"{'strategy':<14}{'avg score':>12}{'best':>10}{'avg moves':>12}{'top tile':>10}{'moves/s':>10}"
    rows = [header, "-" * len(header)]
    for r in sorted(reports, key=lambda r: r.avg_score, reverse=True):
        top_tile = max(r.tile_distribution) if r.games else 0
        rows.append(
            f"{r.strategy:<14}{r.avg_score:>12.0f}{r.best_score:>10}"
            f"{r.avg_moves:>12.0f}{top_tile:>10}{r.moves_per_sec:>10.1f}"
        )
    return "\n".join(rows)
