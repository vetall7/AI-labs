from exceptions import GameplayException
from connect4 import Connect4
from randomagent import RandomAgent
from minmaxagent import MinMaxAgent
from alphabetaagent import AlphaBetaAgent
from minmaxwithoutheu import MinMaxAgentWithoutHeu

connect4 = Connect4(width=7, height=6)
#agent1 = RandomAgent('o')
#agent1 = MinMaxAgentWithoutHeu('x')
agent1 = AlphaBetaAgent('x')
agent2 = MinMaxAgent('o')

while not connect4.game_over:
    connect4.draw()
    try:
        if connect4.who_moves == agent1.my_token:
            n_column = agent1.decide(connect4)
        else:
            n_column = agent2.decide(connect4)
        connect4.drop_token(n_column)
    except (ValueError, GameplayException):
        print('invalid move')

connect4.draw()
