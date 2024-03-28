from copy import copy, deepcopy

from connect4 import Connect4
import math

class MinMaxAgent:
    def __init__(self, token, max_depth=3):
        self.my_token = token
        self.max_depth = max_depth

    def decide(self, game):
        _, column = self.minimax(game, self.max_depth, True)
        return column

    def minimax(self, game, depth, maximizing_player):
        if depth == 0 or game.game_over:
            return self.evaluate(game), None

        if maximizing_player:
            max_eval = -math.inf
            best_column = None
            for column in game.possible_drops():
                game_copy = self.simulate_drop(game, column)
                eval, _ = self.minimax(game_copy, depth - 1, False)
                if eval > max_eval:
                    max_eval = eval
                    best_column = column
            return max_eval, best_column
        else:
            min_eval = math.inf
            best_column = None
            for column in game.possible_drops():
                game_copy = self.simulate_drop(game, column)
                eval, _ = self.minimax(game_copy, depth - 1, True)
                if eval < min_eval:
                    min_eval = eval
                    best_column = column
            return min_eval, best_column

    def evaluate(self, game):
        if game.game_over:
            if game.wins == self.my_token:
                return math.inf
            elif game.wins != None:
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