import math
from copy import copy, deepcopy

from connect4 import Connect4


class AlphaBetaAgent:
    def __init__(self, token, max_depth=3):
        self.my_token = token
        self.max_depth = max_depth

    def decide(self, game):
        _, column = self.alphabeta(game, self.max_depth, -math.inf, math.inf, True)
        return column

    def alphabeta(self, game, depth, alpha, beta, maximizing_player):
        if depth == 0 or game.game_over:
            return self.evaluate(game), None

        if maximizing_player:
            max_eval = -math.inf
            best_column = None
            for column in game.possible_drops():
                game_copy = self.simulate_drop(game, column)
                eval, _ = self.alphabeta(game_copy, depth - 1, alpha, beta, False)
                if eval > max_eval:
                    max_eval = eval
                    best_column = column
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_column
        else:
            min_eval = math.inf
            best_column = None
            for column in game.possible_drops():
                game_copy = self.simulate_drop(game, column)
                eval, _ = self.alphabeta(game_copy, depth - 1, alpha, beta, True)
                if eval < min_eval:
                    min_eval = eval
                    best_column = column
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_column

    def evaluate(self, game):
        if game.game_over:
            if game.wins == self.my_token:
                return math.inf
            elif game.wins is not None:
                return -math.inf
            else:
                return 0

        score = 0
        for four in game.iter_fours():
            if four.count(self.my_token) == 3 and four.count('_') == 1:
                score += 100
            elif four.count(self.my_token) == 2 and four.count('_') == 2:
                score += 10
            elif four.count(self.my_token) == 1 and four.count('_') == 3:
                score += 1

        return score

    def simulate_drop(self, game, column):
        game_copy = deepcopy(game)
        game_copy.drop_token(column)
        return game_copy
