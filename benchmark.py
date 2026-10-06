#!/usr/bin/env python
"""Benchmark 2048 strategies under identical, reproducible conditions.

This is the script behind the "Results" table in the README. It runs each
strategy over the same set of seeded games and reports the metrics that matter
for the goal of *reaching the 2048 tile* (and beyond): average score, best score,
how often the 2048 and 4096 tiles were reached, the highest tile seen, and speed.

    python benchmark.py                                   # the default lineup
    python benchmark.py --strategies all --games 50
    python benchmark.py --strategies expectimax,mcts --depth 4 --games 20
    python benchmark.py --strategies rollouts,mcts --runs 40 --rollout-depth 40
    python benchmark.py --strategies ntuple --ntuple-weights mynet.npz
    python benchmark.py --strategies ntuple --ntuple-untrained   # the raw, unlearned net
    python benchmark.py --strategies ntuple,dqn --lookahead 1    # learned value + search
    python benchmark.py --markdown                        # also print a GitHub table

The trained players (`genetic`, `cmaes`, `qlearning`, `ntuple`, `ppo`, `dqn`,
`imitation`) load the bundled weights by default, so they show what each one can
actually do, not an untrained model.

Both the games *and* each strategy's own RNG are seeded from ``--seed``, so every
strategy faces the same starting boards and the numbers are reproducible run to
run.
"""
import argparse
import itertools
import platform
import time
from pathlib import Path

from playbook.evaluation import evaluate
from playbook.game import SimEnv
from playbook.registry import available

# The trained weights shipped with the repo (see the README for how each was
# trained). Loaded by default so the learners benchmark as *trained* agents.
_LEARNING = Path(__file__).resolve().parent / "playbook" / "strategies" / "learning"
DEFAULT_NTUPLE_WEIGHTS = _LEARNING / "reinforcement" / "ntuple" / "ntuple.npz"
_OPTIMIZATION = Path(__file__).resolve().parent / "playbook" / "strategies" / "optimization"
DEFAULT_WEIGHTS = {
    "genetic": _OPTIMIZATION / "genetic.npy",
    "cmaes": _OPTIMIZATION / "cmaes.npy",
    "dqn": _LEARNING / "reinforcement" / "deep" / "dqn.pt",
    "ppo": _LEARNING / "reinforcement" / "deep" / "ppo.pt",
    "imitation": _LEARNING / "supervised" / "imitation.pt",
    "qlearning": _LEARNING / "reinforcement" / "tabular" / "qlearning.pkl",
}

# Default knobs per strategy when benchmarking toward 2048. Search strategies
# get a depth that is a sensible accuracy/speed trade-off. `rollouts` and `mcts`
# get the same playout budget (20 per move x ~4 moves = 80 simulations), so they
# differ only in where the playouts go. Override search depth with --depth.
PROFILES = {
    "random": {},
    "greedy": {},
    "maximization": {"depth": 3},
    "minimax": {"depth": 3},
    "expectimax": {"depth": 3},
    "rollouts": {"runs": 20, "depth": 20},
    "mcts": {"runs": 80, "depth": 20},
    "ntuple": {},   # weights are resolved in _config (bundled net by default)
    "genetic": {},
    "cmaes": {},
    "ppo": {},
    "qlearning": {},
    "dqn": {},
    "imitation": {},
}

# Sensible "run everything that plays out of the box" lineup. Excludes `manual`
# (needs a human).
DEFAULT_SET = ["random", "greedy", "maximization", "minimax", "expectimax", "rollouts",
               "mcts", "genetic", "cmaes", "qlearning", "ntuple", "ppo", "dqn", "imitation"]


# The neural players need torch (pip install -e ".[deep]").
_NEEDS_TORCH = {"dqn", "ppo", "imitation"}


def _torch_available():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve(names):
    if names == ["all"]:
        if _torch_available():
            return DEFAULT_SET
        print("(torch is not installed: skipping dqn, ppo and imitation)")
        return [n for n in DEFAULT_SET if n not in _NEEDS_TORCH]
    unknown = [n for n in names if n not in available()]
    if unknown:
        raise SystemExit(f"unknown strategy/ies: {unknown}; choices: {available()}")
    return names


# Strategies whose `depth` is the search-tree depth (so --depth applies to them,
# but not to rollouts/mcts, where `depth` is the playout length).
_TREE_SEARCH = {"maximization", "minimax", "expectimax"}
# Playout searchers: --runs and --rollout-depth apply to them.
_PLAYOUTS = {"rollouts", "mcts"}
# Learned afterstate-value players: --lookahead adds expectimax levels on top.
_LOOKAHEAD = {"ntuple", "dqn"}


def _config(name, args):
    cfg = dict(PROFILES.get(name, {}))
    cfg["seed"] = args.seed   # seed every strategy's RNG, not just the games
    if args.depth is not None and name in _TREE_SEARCH:
        cfg["depth"] = args.depth
    if name in _PLAYOUTS:
        if args.runs is not None:
            cfg["runs"] = args.runs
        if args.rollout_depth is not None:
            cfg["depth"] = args.rollout_depth
    if name == "ntuple":
        if args.ntuple_untrained:
            cfg.pop("weights", None)            # play the raw, unlearned net on purpose
        elif args.ntuple_weights:
            cfg["weights"] = args.ntuple_weights
        else:
            cfg["weights"] = str(DEFAULT_NTUPLE_WEIGHTS)   # the bundled trained net
    if name in DEFAULT_WEIGHTS:
        cfg["weights"] = str(DEFAULT_WEIGHTS[name])
    if args.lookahead and name in _LOOKAHEAD:
        cfg["depth"] = args.lookahead
    return cfg


def _label(name, cfg):
    return f"{name} (depth {cfg['depth']})" if name in _LOOKAHEAD and cfg.get("depth") else name


def run(names, games, max_moves, seed, args):
    """Evaluate each strategy over the same seeded games; return (report, config) list."""
    import sys

    from playbook.registry import make_strategy

    out = []
    for name in names:
        cfg = _config(name, args)
        print(f"  running {name} ...", end=" ", flush=True)
        strategy = make_strategy(name, **cfg)
        # A fresh seeded env per game, identical sequence across strategies.
        seeds = itertools.count(seed)
        t0 = time.time()
        report = evaluate(strategy, lambda: SimEnv(seed=next(seeds)),
                          games=games, max_moves=max_moves, name=_label(name, cfg))
        out.append((report, cfg))
        print(f"done ({time.time() - t0:.1f}s)", file=sys.stdout, flush=True)
    return out


def _print_row(report, target):
    print(f"{report.strategy:<20}"
          f"{report.avg_score:>10.0f}"
          f"{report.best_score:>9}"
          f"{100 * report.reach_rate(target):>8.0f}%"
          f"{100 * report.reach_rate(2 * target):>8.0f}%"
          f"{max(report.tile_distribution):>9}"
          f"{report.avg_moves:>9.0f}"
          f"{report.moves_per_sec:>10.1f}")


def _header(target):
    return (f"{'strategy':<20}{'avg':>10}{'best':>9}"
            f"{target:>9}{2 * target:>9}{'top':>9}{'moves':>9}{'moves/s':>10}")


def print_table(results, target):
    print()
    head = _header(target)
    print(head)
    print("-" * len(head))
    for report, _ in sorted(results, key=lambda r: r[0].avg_score, reverse=True):
        _print_row(report, target)


def print_markdown(results, target, games):
    """A GitHub-flavoured table, ready to paste into the README Results section."""
    print(f"\n<!-- {games} games/strategy, seeded, on {platform.processor() or platform.machine()} -->")
    print(f"| Strategy | Avg score | Best | {target} rate | {2 * target} rate | Top tile "
          f"| Avg moves | Moves/s |")
    print("|---|--:|--:|--:|--:|--:|--:|--:|")
    for report, _ in sorted(results, key=lambda r: r[0].avg_score, reverse=True):
        print(f"| `{report.strategy}` "
              f"| {report.avg_score:,.0f} "
              f"| {report.best_score:,} "
              f"| {100 * report.reach_rate(target):.0f}% "
              f"| {100 * report.reach_rate(2 * target):.0f}% "
              f"| {max(report.tile_distribution):,} "
              f"| {report.avg_moves:.0f} "
              f"| {report.moves_per_sec:,.0f} |")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategies", default="all",
                   help="comma-separated names, or 'all' for the default lineup")
    p.add_argument("--games", type=int, default=20, help="games per strategy")
    p.add_argument("--max-moves", dest="max_moves", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0, help="base seed (game i uses seed+i)")
    p.add_argument("--target", type=int, default=2048, help="win-tile for the reach rate")
    p.add_argument("--depth", type=int, default=None,
                   help="override tree-search depth (maximization/minimax/expectimax)")
    p.add_argument("--runs", type=int, default=None,
                   help="playouts per move for rollouts (default %d) / simulations per move "
                        "for mcts (default %d)" % (PROFILES["rollouts"]["runs"],
                                                   PROFILES["mcts"]["runs"]))
    p.add_argument("--rollout-depth", dest="rollout_depth", type=int, default=None,
                   help="playout length for rollouts/mcts (default %d)"
                        % PROFILES["mcts"]["depth"])
    p.add_argument("--ntuple-weights", dest="ntuple_weights", default=None,
                   help="path to a trained n-tuple network (.npz); overrides the bundled one")
    p.add_argument("--ntuple-untrained", dest="ntuple_untrained", action="store_true",
                   help="benchmark the n-tuple net untrained (plays like greedy)")
    p.add_argument("--lookahead", type=int, default=0,
                   help="expectimax levels on top of the learned value (ntuple/dqn)")
    p.add_argument("--markdown", action="store_true", help="also print a README table")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    names = _resolve([s.strip() for s in args.strategies.split(",") if s.strip()])

    print(f"Benchmarking {len(names)} strategies: {', '.join(names)}")
    print(f"{args.games} games each, seed {args.seed}, target {args.target}\n")

    start = time.time()
    results = run(names, args.games, args.max_moves, args.seed, args)
    print_table(results, args.target)
    print(f"\nTotal wall time: {time.time() - start:.1f}s")

    if args.markdown:
        print_markdown(results, args.target, args.games)


if __name__ == "__main__":
    main()
