#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: infer.py
Desc: 模型对战测试
Author:yanjingang(yanjingang@mail.com)
Date: 2019/1/4 22:46
Cmd: python infer.p
"""

from __future__ import print_function
import os
import sys

# PATH
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.realpath(CUR_PATH + '/../../../')
sys.path.append(BASE_PATH)

from machinelearning.lib import logger
from game import Board, Game
from mcts import MCTSPlayer
from net.policy_value_net_keras import PolicyValueNet  # Keras


class HumanPlayer(object):
    """人类玩家"""

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input("Your move: ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "HumanPlayer {}".format(self.player)


def run():
    width, height = 8, 8
    model_file = CUR_PATH+'/model/best_policy_8x8.model'
    try:
        # 初始化棋盘
        board = Board(width=width, height=height, n_in_row=5)
        game = Game(board)

        # 初始化AI棋手
        best_policy = PolicyValueNet(width, height, model_file=model_file)
        """
        # 使用numpy加载训练好的模型(仅限Theano/Lasagne训练出的模型)
        try:
            policy_param = pickle.load(open(model_file, 'rb'))
        except:
            policy_param = pickle.load(open(model_file, 'rb'),
                                       encoding='bytes')  # To support python3
        best_policy = PolicyValueNetNumpy(width, height, policy_param)
        """
        mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=900)
        logger.info("MCTSPlayer: n_playout={}".format(900), MCTSPlayer)

        # 纯MCTS棋手
        # mcts_player = MCTSPurePlayer(c_puct=5, n_playout=4000)

        # 初始化人类棋手，输入移动命令的格式： 2,3
        human_player = HumanPlayer()

        # 启动游戏（start_player=0人类先/1机器先）
        game.start_play(human_player, mcts_player, start_player=1, is_shown=1)

    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
