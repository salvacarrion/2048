import time
import operator

from collections import defaultdict


from browser.chrome.chromectrl import ChromeDebuggerControl
from browser.gamectrl import Fast2048Control
from search import *
from game import *
from utils import get_pretty_time
from heuristics import *


def play_game(gamectrl, heuristic, max_moves=1000, verbose=0):
    num_moves = 0
    start = time.time()
    last_move = '-'
    while num_moves < max_moves:
        # Get game state
        state = gamectrl.get_status()
        if state == 'ended':
            break
        elif state == 'won':
            time.sleep(0.75)
            gamectrl.continue_game()

        # Get board and current score
        board = gamectrl.get_board()
        current_score = gamectrl.get_score()

        # Print data
        if verbose > 0:
            h_score = heuristic.score_hfunc(board)
            print("Time: {:8.3f}\t\tScore: {:08d}\t\th_score: {:08d}\t\tMove: #{:05d} [{:s}]".format(time.time() - start, current_score, h_score, num_moves, last_move))
            print_board(to_val(board))
            print('\n')

        # Compute new move
        move = heuristic.find_best_move(board, current_score)
        if move < 0:
            break

        # Execute move
        gamectrl.execute_move(move)
        last_move = movename(move)
        num_moves += 1

    score = gamectrl.get_score()
    board = gamectrl.get_board()
    maxval = max(max(row) for row in to_val(board))
    print("Game over. Final score %d; highest tile %d." % (score, maxval))
    return score, maxval, num_moves, board


def main(*args):
    #RandomSearch, EagerSearch, MaximizationSearch, MinimaxSearch, ExpectimaxSearch, MonteCarloSearch
    PORT = 9222
    DELAY = None
    NUM_TESTS = 10
    MAX_MOVES = 2000
    MAX_DEPTH = 5  #Possible combinations 4^depth
    RUNS = 30  # Size of population
    ADV_SEARCH = MinimaxSearch  # Cost= 4*runs*depth
    HEURISTIC = h_aprox_score

    # Setup game controler in Chrome
    ctrl = ChromeDebuggerControl(PORT)
    gamectrl = Fast2048Control(ctrl)

    # Define heuristic
    cHeuristic = ADV_SEARCH(delay=DELAY, max_depth=MAX_DEPTH, runs=RUNS, score_hfunc=HEURISTIC, verbose=False)

    print("\nGaming rounds:")
    results = []
    start_time = time.time()
    for i in range(NUM_TESTS):
        print("\t#{}. Playing new game...\n".format(i + 1))

        # Restart game
        gamectrl.restart_game()

        res = play_game(gamectrl, cHeuristic, max_moves=MAX_MOVES, verbose=1)
        results.append(res)
    end_time = time.time()

    print("")
    print("")
    print("-----------------------------------------------------------------")
    print("Parameters:")
    print("\tMax. moves: {:d}".format(MAX_MOVES))
    print("\tMax. depth: {:d}".format(MAX_DEPTH))
    print("\tRuns: {:d}".format(RUNS))
    print("")
    print("Final stats:")
    d_maxtiles = defaultdict(int)
    avg_score = 0
    avg_moves = 0
    most_freq_tile = None
    for i, t in enumerate(results):
        score, maxval, num_moves, board = t
        d_maxtiles[maxval] += 1
        avg_score += score
        avg_moves += num_moves
        print(
            '\t#{:<3}\tScore: {:<6}\t\tMax. tile: {:<5}\t\tTotal moves: #{:<5}\t\tMax. depth: {:d}'.format(i + 1, score,
                                                                                                           maxval,
                                                                                                           num_moves,
                                                                                                           cHeuristic.max_depth))

    # Print stats
    elapsed_time = get_pretty_time(start_time, end_time, divisor=NUM_TESTS, max_decimals=5)
    most_freq_tile = max(d_maxtiles.items(), key=operator.itemgetter(1))[0]
    avg_moves = avg_moves / NUM_TESTS
    print('\n\t(*)\t\tAvg. score: {:<6}\tMost freq. tile: {:<5}\tAvg. moves: #{:<5}\t\tAvg. depth: {:<3}'.format(
        int(avg_score / NUM_TESTS),
        most_freq_tile,
        int(avg_moves),
        cHeuristic.max_depth))
    print('\t(*)\t\tAvg. time: {:s}  ({:.2f} moves/s)'.format(elapsed_time, avg_moves / (end_time - start_time)))

    print("\nTop tiles: ")
    sorted_tiles = [(k, v) for k, v in d_maxtiles.items()]
    sorted_tiles = sorted(sorted_tiles, key=lambda x: x[0], reverse=True)
    for i, t in enumerate(sorted_tiles):
        k, v = t
        print('\t{:d}.\tTile {:d} was reached {:d} times ({:.2f}%)'.format(i + 1, k, v, v / NUM_TESTS * 100))

    print("")
    print("Total time: {}".format(get_pretty_time(start_time, end_time)))
    return -avg_score


if __name__ == "__main__":
    main()
    # import cma
    # es = cma.fmin2(main, [1.0] * 3, 0.5)
    # print(es)

