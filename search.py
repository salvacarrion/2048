import random
import time
import numpy as np
import sys
from game import *
from copy import copy, deepcopy


class BaseSearch:

    def __init__(self, delay=None, max_depth=3, runs=1, score_hfunc=None, verbose=False):
        self.delay = delay
        self.max_depth = max_depth
        self.score_hfunc = score_hfunc
        self.verbose = verbose
        self.runs = runs

    def score_func(self, *args, **kwargs):
        return self.score_hfunc(*args, **kwargs)

    def wait(self):
        if self.delay:
            time.sleep(self.delay)

    def move(self, move, board, score):
        # if board:
        #     sc = int(np.sum(self.score_func(board)))
        #     print("\t=> Current score: {}\n".format(sc))

        new_board, new_score = make_move(board, move)
        new_score = score_to_val(new_score)

        if self.verbose:
            print_board(to_val(new_board))
            print('New score: {} + {} = {}'.format(score, new_score, score + new_score))
            print('=============================================')

        self.wait()
        return move


class ManualSearch(BaseSearch):

    def find_best_move(self, board=None, score=None):
        move = int(input("Select move: "))
        self.wait()
        return move


class RandomSearch(BaseSearch):

    """
    Move randomly
    """
    def find_best_move(self, board=None, score=None):
        move = random.randint(0, 3)
        return super().move(move, board, score)


class EagerSearch(BaseSearch):

    """
    Make the move tha give you the highest inmediate reward
    """
    def find_best_move(self, board=None, score=None):

        max_score = 0
        move = random.randint(0, 3)
        for m in range(4):  # up, down, left, right
            new_board, new_score = make_move(board, m)
            new_score = score_to_val(new_score)

            if new_score > max_score:
                max_score = new_score
                move = m

        return super().move(move, board, score)


class MaximizationSearch(BaseSearch):
    """
    Maximize the score planning d steps ahead. Se random tile, randomly
    """
    def maximize(self, board, depth):
        if depth == 0 or is_blocked(board):
            return self.score_func(board)

        best_score = 0
        for m in range(4):  # up, down, left, right
            # Make move and add random tile
            new_board, new_score = make_move(board, m)
            add_random_tile(new_board)

            move_score = self.maximize(new_board, depth - 1)
            best_score = max(best_score, move_score)

        return best_score

    def find_best_move(self, board=None, score=None):

        move = random.randint(0, 3)
        best_score = 0
        valid_moves = get_valid_moves(board)

        # Maximization (get the best move)
        for m in range(4):  # up, down, left, right
            if m not in valid_moves:  # Search only valid moves
                continue

            # Make move and add random tile
            new_board, new_score = make_move(board, m)
            add_random_tile(new_board)

            # Get best move
            move_score = self.maximize(new_board, self.max_depth)
            if move_score > best_score:
                best_score = move_score
                move = m

        return super().move(move, board, score)


class MinimaxSearch(BaseSearch):
    """
    Maximize the score, minimize the chance of winning by placing the random tile in the worst position
    """
    def minimax(self, board, depth, alpha, beta, is_user):
        if depth == 0 or is_blocked(board):
            return self.score_func(board)

        if is_user:  # Plays the user [Maximization]
            best_score = -sys.maxsize - 1
            for m in range(4):  # up, down, left, right
                # Score move
                new_board, new_score = make_move(board, m)
                move_score = self.minimax(new_board, depth - 1, alpha, beta, is_user=False)
                best_score = max(best_score, move_score)

                # Prunning
                alpha = max(alpha, best_score)
                if alpha >= beta:
                    break
            return best_score

        else:  # Plays the machine [Minimization]
            free_tiles = get_free_tiles(board)

            # If there are no empty tiles, then nothing
            if not len(free_tiles):
                return self.score_func(board)

            worst_score = sys.maxsize
            for i, j in free_tiles:
                new_board = set_tile(board, i, j)
                tile_score = self.minimax(new_board, depth - 1, alpha, beta, is_user=True)
                worst_score = min(worst_score, tile_score)

                # Pruning
                beta = min(beta, worst_score)
                if alpha >= beta:
                    break
            return worst_score

    def find_best_move(self, board=None, score=None):

        move = random.randint(0, 3)
        best_score = 0
        valid_moves = get_valid_moves(board)

        # Maximization (get the best move)
        for m in range(4):  # up, down, left, right
            if m not in valid_moves:  # Search only valid moves
                continue

            # Score move
            new_board, new_score = make_move(board, m)
            move_score = self.minimax(new_board, self.max_depth, alpha=-sys.maxsize - 1, beta=sys.maxsize, is_user=False)

            if move_score > best_score:
                best_score = move_score
                move = m

        return super().move(move, board, score)


class ExpectimaxSearch(BaseSearch):

    def expectimax(self, board, depth, is_user):
        if depth == 0 or is_blocked(board):
            return self.score_func(board)

        if is_user:  # Plays the user [Maximization]
            valid_moves = get_valid_moves(board)
            best_score = -sys.maxsize - 1
            for m in range(4):  # up, down, left, right
                if m not in valid_moves:  # Search only valid moves
                    continue

                # Score move
                new_board, new_score = make_move(board, m)
                move_score = self.expectimax(new_board, depth - 1, is_user=False)
                best_score = max(best_score, move_score)
            return best_score

        else:  # Plays the machine [Minimization]
            free_tiles = get_free_tiles(board)

            # If there are no empty tiles, then nothing
            if not len(free_tiles):
                return self.score_func(board)

            tile_score = 0
            for i, j in free_tiles:
                # Prob 2
                new_board = set_tile(board, i, j, value=2)
                tile_score += 0.9 * self.expectimax(new_board, depth-1, is_user=True)

                # Prob 4
                new_board = set_tile(board, i, j, value=4)
                tile_score += 0.1 * self.expectimax(new_board, depth - 1, is_user=True)
            tile_score /= len(free_tiles)  # Normalize score (probability)
            return tile_score

    def find_best_move(self, board=None, score=None):

        move = random.randint(0, 3)
        best_score = 0
        valid_moves = get_valid_moves(board)

        # Maximization (get the best move)
        for m in range(4):  # up, down, left, right
            if m not in valid_moves:  # Search only valid moves
                continue

            # Score move
            new_board, new_score = make_move(board, m)
            move_score = self.expectimax(new_board, self.max_depth, is_user=False)

            if move_score > best_score:
                best_score = move_score
                move = m

        return super().move(move, board, score)


class MonteCarloSearch(BaseSearch):

    def mc(self, board, depth):
        new_board = board

        for i in range(depth):
            if is_blocked(board):  # If there are no moves left
                break
            # Make random move and add tile
            move = random.randint(0, 3)
            new_board, new_score = make_move(new_board, move)
            add_random_tile(new_board)

        return self.score_func(new_board)

    def find_best_move(self, board=None, score=None):

        valid_moves = get_valid_moves(board)

        scores = {0: [], 1: [], 2: [], 3: []}
        # Maximization (get the best move)
        for m in range(4):  # up, down, left, right
            if m not in valid_moves:  # Search only valid moves
                continue

            # Make move and add tile
            new_board, new_score = make_move(board, m)
            add_random_tile(new_board)

            # Test with n runs
            for i in range(self.runs):
                score = self.mc(new_board, self.max_depth)
                scores[m].append(score)

        # Compute average score
        up_avg_score = sum(scores[0])/len(scores[0]) if len(scores[0]) else 0
        down_avg_score = sum(scores[1])/len(scores[1]) if len(scores[1]) else 0
        left_avg_score = sum(scores[2])/len(scores[2]) if len(scores[2]) else 0
        right_avg_score = sum(scores[3])/len(scores[3]) if len(scores[3]) else 0

        # Get highest avg score
        scores = [up_avg_score, down_avg_score, left_avg_score, right_avg_score]
        move = scores.index(max(scores))

        return super().move(move, board, score)


