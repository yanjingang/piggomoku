#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: test.py
Desc: mcts蒙特卡罗树搜索测试
Author:yanjingang(yanjingang@mail.com)
Date: 2019/1/5 19:46
Cmd: python test.py
"""

from __future__ import print_function
import os
import sys
import time
import random
import numpy as np
from collections import defaultdict, deque

# PATH
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.realpath(CUR_PATH + '/../../../')
sys.path.append(BASE_PATH)
# print(CUR_PATH, BASE_PATH)

from machinelearning.lib import logger
from game import Board
#from mcts_pure import MCTSPlayer as MCTSPurePlayer
#from mcts_pure import MCTS
from mcts import MCTSPlayer
# from net.policy_value_net import PolicyValueNet  # Theano and Lasagne
# from net.policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from net.policy_value_net_tensorflow import PolicyValueNet # Tensorflow
from net.policy_value_net_keras import PolicyValueNet  # Keras


class Game(object):
    """游戏对局"""

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        """绘制棋盘并显示游戏信息"""
        width = board.width
        height = board.height

        print("Player", player1, "with X")
        print("Player", player2, "with O")
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                p = board.states.get(i * width + j, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """启动游戏"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) or 1 (player2 first)')
        # 初始化棋盘
        self.board.init_board(start_player)
        # 指定对局玩家
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        # 绘制棋盘
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        # 开始对局
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            # 获取落子位置并落子
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            # 检查游戏是否结束
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=1, temp=1e-3):
        """
        使用MCTS蒙特卡罗树搜索进行自我对抗
            重用搜索树并保存自我对抗数据用于训练(state, mcts_probs, winners_z)
            player：mcts_player
        """
        # 初始化棋盘
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:  # 在棋局没有赢家或和棋结束前交替落子
            # MCTS搜索最佳落子位置
            print('------get_action------')
            move, move_probs = player.get_action(self.board, temp=temp, return_prob=1)
            print(move)
            print(move_probs)
            # 保存当前盘面
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # 执行落子
            print('------do_move------')
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            # 检查游戏是否结束
            print('------check_game_end------')
            end, winner = self.board.game_end()
            if end:
                # 从当前玩家视角确定winner
                winners_z = np.zeros(len(current_players))
                if winner != -1:  # 不是和棋
                    winners_z[np.array(current_players) == winner] = 1.0  # 更新赢家步骤位置=1
                    winners_z[np.array(current_players) != winner] = -1.0  # 更新输家步骤位置=-1
                # 重置MCTS根结点
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
            #
            time.sleep(10)



import copy
from operator import itemgetter


def rollout_policy_fn(board):
    """给棋盘所有可落子位置随机分配概率"""
    action_probs = np.random.rand(len(board.availables))
    return zip(board.availables, action_probs)


def policy_value_fn(board):
    """给棋盘所有可落子位置分配默认平均概率 [(0, 0.015625), (action, probability), ...], 0"""
    action_probs = np.ones(len(board.availables))/len(board.availables)
    print("__policy_value_fn__")
    print(len(board.availables),action_probs)
    return zip(board.availables, action_probs), 0


class TreeNode(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """A simple implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        print('XXXXXXXXXXXXXXXXXXXXXXXXX-0')
        node = self._root
        while(1):
            if node.is_leaf():
                print("node.is_leaf")
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            print('-------node.select--------')
            print(action,node._children.items())
            state.do_move(action)

        print('XXXXXXXXXXXXXXXXXXXXXXXXX-a')
        action_probs, _ = self._policy(state)
        print(action_probs, _ )
        print('XXXXXXXXXXXXXXXXXXXXXXXXX-b')
        # Check for end of game
        end, winner = state.game_end()
        print(end,winner)
        if not end:
            node.expand(action_probs)
	    
        print('XXXXXXXXXXXXXXXXXXXXXXXXX-c')
        # Evaluate the leaf node by random rollout
        leaf_value = self._evaluate_rollout(state)
        print(leaf_value)
        print('XXXXXXXXXXXXXXXXXXXXXXXXX-d')
        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def _evaluate_rollout(self, state, limit=1000):
        """Use the rollout policy to play until the end of the game,
        returning +1 if the current player wins, -1 if the opponent wins,
        and 0 if it is a tie.
        """
        print('------_evaluate_rollout-------')
        player = state.get_current_player()
        for i in range(limit):
            end, winner = state.game_end()
            if end:
                print("board.game_end")
                break
            action_probs = rollout_policy_fn(state)
            #print(action_probs)
            max_action = max(action_probs, key=itemgetter(1))[0]
            state.do_move(max_action)
        else:
            # If no break from the loop, issue a warning.
            print("WARNING: rollout reached move limit")
        if winner == -1:  # tie
            return 0
        else:
            return 1 if winner == player else -1

    def get_move(self, state):
        """Runs all playouts sequentially and returns the most visited action.
        state: the current game state

        Return: the selected action
        """
        print("__get_move__")
        print(state)
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        return max(self._root._children.items(),
                   key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"

class MCTSPurePlayer(object):
    """AI player based on MCTS"""
    def __init__(self, c_puct=5, n_playout=2000):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board):
        sensible_moves = board.availables
        print('--------board.availables-------')
        print(sensible_moves)
        if len(sensible_moves) > 0:
            move = self.mcts.get_move(board)
            print('-------self.mcts.get_move--------')
            print(move)
            self.mcts.update_with_move(-1)
            return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)


"""
def rollout_policy_fn(board):
    # rollout randomly
    action_probs = np.random.rand(len(board.availables))
    return zip(board.availables, action_probs)


def policy_value_fn(board):
    # return uniform probabilities and 0 score for pure MCTS
    action_probs = np.ones(len(board.availables))/len(board.availables)
    return zip(board.availables, action_probs), 0
"""


class MctsTest():
    def __init__(self, init_model=None):
        # 棋盘大小 8*8, 5个子连起来
        self.board_width = 8
        self.board_height = 8
        self.n_in_row = 5  # n子相连
        self.policy_evaluate_size = 2  # 策略评估胜率时的模拟对局次数
        self.batch_size = 1  # data_buffer中对战次数超过n次后开始启动模型训练
        self.board = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
        self.game = Game(self.board)
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # 基于KL的自适应学习率
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # 每个动作的模拟次数
        self.c_puct = 5
        self.buffer_size = 10000  # cache对战记录个数
        self.data_buffer = deque(maxlen=self.buffer_size)  # 完整对战历史记录，用于训练
        self.epochs = 5  # 每次更新策略价值网络的训练步骤数
        self.kl_targ = 0.02  # 策略价值网络KL值目标
        self.best_win_ratio = 0.0
        # 纯MCT的模拟数，用于评估策略模型
        self.pure_mcts_playout_num = 5
        self.policy_value_net = PolicyValueNet(self.board_width, self.board_height)
        # 创建使用策略价值网络来指导树搜索和评估叶节点的MCTS玩家
        """self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)"""

    def get_equi_data(self, play_data):
        """
        通过旋转和翻转增加数据集
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # 逆时针旋转
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
                # 水平翻转
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
        return extend_data

    def policy_update(self):
        """更新策略价值网络policy-value"""
        # 随机抽取data_buffer中的对抗数据
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        # 训练策略价值网络
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate * self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                                axis=1)
                         )
            if kl > self.kl_targ * 4:  # 如果D_KL跑偏则尽早停止
                break
        # 自动调整学习率
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        logger.info(("TEST kl:{:.5f},"
                     "lr_multiplier:{:.3f},"
                     "loss:{},"
                     "entropy:{},"
                     "explained_var_old:{:.3f},"
                     "explained_var_new:{:.3f}"
                     ).format(kl,
                              self.lr_multiplier,
                              loss,
                              entropy,
                              explained_var_old,
                              explained_var_new), MctsTest)
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """
        策略胜率评估：模型与纯MCTS玩家对战n局看胜率
        """
        # AlphaGo Zero风格的MCTS玩家（使用策略价值网络来指导树搜索和评估叶节点）
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        # 纯MCTS玩家
        pure_mcts_player = MCTSPurePlayer(c_puct=5, n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            # 对战
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            win_cnt[winner] += 1
        # 胜率
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        logger.info("TEST Num_playouts:{}, win: {}, lose: {}, tie:{}".format(self.pure_mcts_playout_num,
                                                                        win_cnt[1], win_cnt[2], win_cnt[-1]), MctsTest)
        return win_ratio


    def run(self):
        """启动训练"""
        try:

            #test
            # 初始化棋盘
            self.board.init_board()
            print(self.board)
            print(self.board.current_player)
            print(self.board.availables)
            print(self.board.states)
            print(self.board.last_move)

            p1, p2 = self.board.players
            states, mcts_probs, current_players = [], [], []
            # 纯MCTS玩家
	    #player = self.mcts_player
            player = MCTSPurePlayer(c_puct=5, n_playout=self.pure_mcts_playout_num)
            print('------get_action------')
            #move, move_probs = player.get_action(self.board, temp=self.temp, return_prob=1)
            move = player.get_action(self.board)
            print(move)
            """# 保存当前盘面
            states.append(self.board.current_state())
            current_players.append(self.board.current_player)
            # 执行落子
            print('------do_move------')
            self.board.do_move(move)
            self.game.graphic(self.board, p1, p2)
            # 检查游戏是否结束
            print('------check_game_end------')
            end, winner = self.board.game_end()
            if end:
                # 从当前玩家视角确定winner
                winners_z = np.zeros(len(current_players))
                if winner != -1:  # 不是和棋
                    winners_z[np.array(current_players) == winner] = 1.0  # 更新赢家步骤位置=1
                    winners_z[np.array(current_players) != winner] = -1.0  # 更新输家步骤位置=-1
                # 重置MCTS根结点
                player.reset_player()
                if winner != -1:
                    print("Game end. Winner is player:", winner)
                else:
                    print("Game end. Tie")
                print(winner, zip(states, mcts_probs, winners_z))
"""

            """
            i=0
            # 1.收集自我对抗数据
            # 使用MCTS蒙特卡罗树搜索进行自我对抗
            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            print(play_data)
            print(self.episode_len)
            # 把翻转棋盘数据加到数据集里
            play_data = self.get_equi_data(play_data)
            # 保存对抗数据到data_buffer
            self.data_buffer.extend(play_data)
            logger.info("TEST Batch i:{}, episode_len:{}".format(i + 1, self.episode_len), MctsTest)

            # 2.使用对抗数据重新训练策略价值网络模型
            if len(self.data_buffer) >= self.batch_size:
                loss, entropy = self.policy_update()

            # 3.检查一下当前模型胜率
            logger.info("TEST Current self-play batch: {}".format(i + 1), MctsTest)
            # 策略胜率评估：模型与纯MCTS玩家对战n局看胜率
            win_ratio = self.policy_evaluate(self.policy_evaluate_size)
            self.policy_value_net.save_model(CUR_PATH + '/model/current_test_{}_{}.model'.format(self.board_width, self.board_height))
            if win_ratio > self.best_win_ratio:  # 胜率超过历史最优模型
                logger.info("TEST New best policy!!!!!!!!batch:{} win_ratio:{}->{} pure_mcts_playout_num:{}".format(i + 1, self.best_win_ratio, win_ratio, self.pure_mcts_playout_num),
                            MctsTest)
                self.best_win_ratio = win_ratio
                # 保存当前模型为最优模型best_policy
                self.policy_value_net.save_model(CUR_PATH + '/model/best_test_{}_{}.model'.format(self.board_width, self.board_height))
                # 如果胜率=100%，则增加纯MCT的模拟数
                if (self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 5000):
                    self.pure_mcts_playout_num += 1000
                    self.best_win_ratio = 0.0
            """
        except KeyboardInterrupt:
            logger.info('\n\rquit', MctsTest)


if __name__ == '__main__':
    #model_file = CUR_PATH + '/model/best_policy_8_8_keras.model'
    #logger.debug("init") 
    #policy_value_net = PolicyValueNet(8, 8, model_file=model_file)
    #logger.debug("done") 

    ut = MctsTest()
    ut.run()

