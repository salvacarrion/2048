# Playbook — strategies for 2048

![python](https://img.shields.io/badge/python-3.9%2B-blue)
![strategies](https://img.shields.io/badge/strategies-12-orange)
![focus](https://img.shields.io/badge/focus-didactic-success)

![2048 game](images/2048.jpg)

A didactic catalog of AI strategies for [2048](https://play2048.co), built so you can **try a new idea and benchmark it with a single command** — without reading the whole codebase. Every strategy is a small, self-contained file grouped by technique into recognizable "chapters": baselines, search, optimization and (reinforcement) learning. A fast in-memory simulator drives training and evaluation; the exact same strategy can also play a live game in Chrome over the DevTools protocol.

> Clarity over raw performance. The point is to make it obvious *where* an idea lives and *how* to add your own.

## Highlights

- **One interface for every player** — `select_move(board, legal) -> Move`. That is the whole contract.
- **Strategies grouped by technique** — search (minimax, expectimax, MCTS…), optimization (genetic), reinforcement learning (n-tuple TD, DQN), supervised (imitation).
- **Reusable heuristics** — monotonicity, corner gradients, free tiles, merges… combined with explicit weights, no hidden globals.
- **Simulator *or* live browser** behind the same `Env` API, so a strategy you trained offline can play the real game unchanged.
- **Reproducible benchmarks** — one script runs any subset of strategies over the same seeded games and prints a table (and a Markdown table for this README).
- **Pinned mechanics** — the board engine is frozen against a golden fixture of 1600 transitions, so refactors can't silently change the game.

## Install

```bash
pip install -e .             # core (numpy only)
pip install -e ".[browser]"  # + play the live game in Chrome (websocket-client)
pip install -e ".[deep]"     # + deep-RL strategies (torch)
pip install -e ".[dev]"      # + pytest
```

## Quick start

```bash
python -m playbook list                                            # every strategy
python -m playbook compare --strategies random,greedy,expectimax   # head-to-head table
python -m playbook eval  --strategy expectimax --depth 3 --games 10
python -m playbook train --strategy ntuple --episodes 20000 --save ntuple.npz
python -m playbook eval  --strategy ntuple --weights ntuple.npz --games 50
python -m playbook play  --strategy mcts --env browser             # live, in Chrome
```

To play the live game, start Chrome with remote debugging and open the board in that window:

```bash
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222
# then navigate to https://play2048.co
```

## Strategies

| Strategy | Family | Idea | Reaches 2048? |
|---|---|---|:--:|
| `random` | baseline | pick any legal move — the floor everything beats | ✗ |
| `greedy` | baseline | the move with the best immediate score | ✗ |
| `manual` | baseline | you type the moves (for debugging / playing) | depends on you |
| `maximization` | search | look ahead, assume an average random spawn | ✓ |
| `minimax` | search | α-β, treats the spawn as an adversary | ✓ |
| `expectimax` | search | α-β over the *real* 2/4 spawn distribution — the classic strong baseline | ✓✓ |
| `mcts` | search | random rollouts from each candidate move | ✓ |
| `genetic` | optimization | evolve the weights of a heuristic player | ✓ |
| `ntuple` | reinforcement (TD) | learn a value function over tile patterns, no neural net | ✓✓✓ |
| `dqn` | reinforcement (deep) | deep Q-network *(scaffold — interface ready, training is your exercise)* | — |
| `qlearning` | reinforcement (tabular) | classic tabular Q-learning *(scaffold)* | — |
| `imitation` | supervised | learn from a teacher's games *(scaffold)* | — |

Search strategies take a `--depth` (and `--runs` for MCTS) and an injectable `--heuristic`. The `ntuple` agent is the recommended entry point into RL: it learns strong play on a CPU in minutes, with no neural network, and reliably reaches 2048 once trained.

## Results

Reproduce on your machine — every strategy plays the same seeded games:

```bash
python benchmark.py --strategies all --games 50 --markdown
```

Example run (20 games, seed 0; search at depth 3; `ntuple` trained 20k episodes). Numbers vary by machine and seed — regenerate with the command above.

| Strategy | Avg score | Best | 2048 rate | Top tile | Moves/s |
|---|--:|--:|--:|--:|--:|
| `ntuple` (trained) | 41,000 | 73,000 | 84% | 4,096 | 4,700 |
| `expectimax` | 25,000 | 47,000 | 62% | 2,048 | 40 |
| `mcts` | 13,000 | 22,000 | 14% | 2,048 | 90 |
| `genetic` | 9,000 | 16,000 | 4% | 1,024 | 3,600 |
| `greedy` | 4,000 | 6,400 | 0% | 512 | 6,400 |
| `random` | 1,000 | 1,500 | 0% | 128 | 6,600 |

Read the table as two questions: *how strong* (avg/best score, 2048 rate, top tile) and *how cheap* (moves/s). `greedy` and `random` are essentially free but plateau early; `expectimax` is strong but pays per move for its lookahead; `ntuple` does its expensive work once during training and then plays both strongly and fast.

## How it works

Three decoupled layers with a registry and CLI on top:

```text
playbook/
  game/         the world: board mechanics, rules, and the Env (sim + browser)
  heuristics/   reusable board-evaluation functions (the shared "ideas")
  strategies/   the players, grouped by technique:
      baselines/      random · greedy · manual
      search/         maximization · minimax · expectimax · mcts
      optimization/   genetic   (+ room for cma-es, annealing, hill-climb)
      learning/
          supervised/     imitation
          reinforcement/  tabular · ntuple (worked) · deep (dqn)
  evaluation/   play games, aggregate metrics, compare strategies
  registry.py   name -> strategy factory
  cli.py        the commands shown above
benchmark.py    the Results table above
```

Boards are 4×4 numpy arrays of log2 exponents (`0`=empty, `1`=tile 2, …, `11`=2048). The `Env` is the seam that lets one strategy run against the fast simulator or a live Chrome tab without knowing which. Search strategies receive a heuristic and never know which one; learning strategies add a `Trainable` mixin (`train` / `observe` / `save` / `load`) and learn on *afterstates* — the board after the slide+merge but before the random spawn.

## Create your own strategy

Adding an idea is three steps and never touches the engine:

1. **Copy the closest file** in the relevant family (e.g. [`strategies/search/expectimax.py`](playbook/strategies/search/expectimax.py)) and rewrite the one method that matters:

   ```python
   from playbook.strategies.base import Strategy

   class CornerStrategy(Strategy):
       name = "corner"
       def select_move(self, board, legal):
           # board: 4x4 log2 exponents; legal: set[Move]; return a Move.
           return max(legal)   # your idea here
   ```

2. **Register a name** — one line in [`registry.py`](playbook/registry.py) (use a local import so optional deps load only when requested):

   ```python
   reg["corner"] = lambda **c: CornerStrategy(**c)
   ```

3. **Run it** — `python -m playbook eval --strategy corner --games 20`, then add it to the benchmark and compare.

If your strategy *learns*, also mix in `Trainable` and implement `observe` / `train` / `save` / `load`; the `ntuple` agent in [`strategies/learning/reinforcement/ntuple/`](playbook/strategies/learning/reinforcement/ntuple/ntuple.py) is a complete worked example. Each family folder has a `README.md` describing the technique and what to keep in mind.

## Tests

```bash
pytest
```

The board mechanics are pinned to the original game by a golden fixture (`tests/fixtures/moves.json`): 400 seeded boards × 4 moves, regenerated from the pre-refactor engine. If a change to the board silently alters the game, these fail.

## References

- [What is the optimal algorithm for the game 2048?](https://stackoverflow.com/questions/22342854/what-is-the-optimal-algorithm-for-the-game-2048) — the canonical discussion of expectimax + heuristics.
- Szubert & Jaśkowski, *Temporal Difference Learning of N-Tuple Networks for the Game 2048* (2014) — the basis for the `ntuple` strategy.
- [nneonneo/2048-ai](https://github.com/nneonneo/2048-ai) — a fast C++ expectimax implementation.
- [How an AI crushed all human 2048 records](http://www.randalolson.com/2015/04/27/artificial-intelligence-has-crushed-all-human-records-in-2048-heres-how-the-ai-pulled-it-off/) and [the MDP view of 2048](https://jdlm.info/articles/2018/03/18/markov-decision-process-2048.html).
