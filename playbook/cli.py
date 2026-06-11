"""Command-line entry point: try and compare strategies without editing code.

    python -m playbook list
    python -m playbook eval    --strategy expectimax --depth 3 --games 10
    python -m playbook compare --strategies random,greedy,expectimax --games 20
    python -m playbook play     --strategy mcts --env browser
    python -m playbook train    --strategy ntuple --episodes 20000 --save ntuple.npz
"""
import argparse

from .evaluation.compare import compare
from .evaluation.evaluate import evaluate
from .evaluation.report import compare_table
from .evaluation.runner import play_game
from .game.env import BrowserEnv, SimEnv
from .registry import available, make_strategy


def _env_factory(args):
    if args.env == "browser":
        return lambda: BrowserEnv(port=args.port)
    return lambda: SimEnv()


def _strategy_config(args):
    """Collect the optional knobs into a config dict (only those provided)."""
    config = {}
    for key in ("depth", "runs", "heuristic", "weights"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    return config


def cmd_list(args):
    print("Available strategies:")
    for name in available():
        print(f"  {name}")


def cmd_eval(args):
    strategy = make_strategy(args.strategy, **_strategy_config(args))
    report = evaluate(strategy, _env_factory(args), games=args.games,
                      max_moves=args.max_moves, name=args.strategy,
                      render_every=args.render_every)
    print(report.summary())


def cmd_compare(args):
    names = [s.strip() for s in args.strategies.split(",") if s.strip()]
    reports = compare(names, _env_factory(args), games=args.games, max_moves=args.max_moves)
    print(compare_table(reports))


def cmd_play(args):
    strategy = make_strategy(args.strategy, **_strategy_config(args))
    env = _env_factory(args)()
    result = play_game(strategy, env, max_moves=args.max_moves,
                       render_every=args.render_every or 1)
    print(f"\nGame over — score {result.score}, top tile {result.max_tile}, "
          f"{result.moves} moves")


def cmd_train(args):
    strategy = make_strategy(args.strategy, **_strategy_config(args))
    if not hasattr(strategy, "train"):
        raise SystemExit(f"{args.strategy!r} is not trainable")
    strategy.train(SimEnv(), episodes=args.episodes, verbose=True)
    if args.save:
        strategy.save(args.save)
        print(f"Saved to {args.save}")


def build_parser():
    p = argparse.ArgumentParser(prog="playbook", description="Strategies for 2048.")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--env", choices=["sim", "browser"], default="sim")
        sp.add_argument("--port", type=int, default=9222)
        sp.add_argument("--games", type=int, default=10)
        sp.add_argument("--max-moves", dest="max_moves", type=int, default=10_000)
        sp.add_argument("--render-every", dest="render_every", type=int, default=0)

    def add_strategy_knobs(sp):
        sp.add_argument("--strategy", required=True)
        sp.add_argument("--depth", type=int)
        sp.add_argument("--runs", type=int)
        sp.add_argument("--heuristic")
        sp.add_argument("--weights")

    sp = sub.add_parser("list", help="list available strategies")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("eval", help="evaluate one strategy")
    add_common(sp); add_strategy_knobs(sp); sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("compare", help="compare several strategies")
    add_common(sp)
    sp.add_argument("--strategies", required=True, help="comma-separated names")
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("play", help="play a single (optionally live) game")
    add_common(sp); add_strategy_knobs(sp); sp.set_defaults(func=cmd_play)

    sp = sub.add_parser("train", help="train a learning strategy")
    add_strategy_knobs(sp)
    sp.add_argument("--episodes", type=int, default=10_000)
    sp.add_argument("--save")
    sp.set_defaults(func=cmd_train)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
