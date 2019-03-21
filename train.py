#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: infer.py
Desc: 五子棋强化学习模型训练
Author:yanjingang(yanjingang@mail.com)
Date: 2019/1/4 22:46
Cmd: nohup python train.py >log/train.log &
"""

from __future__ import print_function
import os
import sys
import random
import numpy as np
from collections import defaultdict, deque

# PATH
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.realpath(CUR_PATH + '/../../../')
sys.path.append(BASE_PATH)

from machinelearning.lib import logger
from game import Board, Game
from mcts import MCTSPurePlayer, MCTSPlayer
from net.policy_value_net_keras import PolicyValueNet  # Keras
# from net.policy_value_net import PolicyValueNet  # Theano and Lasagne
# from net.policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from net.policy_value_net_tensorflow import PolicyValueNet # Tensorflow


class GomokuTrainPipeline():
    def __init__(self, init_model=None):
        # 棋盘大小 8*8, 5个子连起来
        self.board_width = 8
        self.board_height = 8
        self.n_in_row = 5  # n子相连
        self.policy_evaluate_size = 10  # 策略评估胜率时的模拟对局次数
        self.game_batch_num = 10000  # selfplay对战次数
        self.batch_size = 512  # data_buffer中对战次数超过n次后开始启动模型训练
        self.check_freq = 50  # 每对战n次检查一次当前模型vs旧模型胜率
        self.board = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
        self.game = Game(self.board)
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # 基于KL的自适应学习率
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # 每个动作的模拟次数
        self.buffer_size = 10000  # cache对战记录个数
        self.data_buffer = deque(maxlen=self.buffer_size)  # 完整对战历史记录，用于训练
        self.play_batch_size = 1
        self.epochs = 5  # 每次更新策略价值网络的训练步骤数
        self.kl_targ = 0.02  # 策略价值网络KL值目标
        self.best_win_ratio = 0.0
        # 纯MCTS的模拟数，用于评估策略模型
        self.pure_mcts_playout_num = 1000  # 用户纯MCTS构建初始树时的随机走子步数
        self.c_puct = 5  # MCTS child搜索深度
        if init_model:
            # 使用一个训练好的策略价值网络
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height, model_file=init_model)
        else:
            # 使用一个新的的策略价值网络
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height)
        # 创建使用策略价值网络来指导树搜索和评估叶节点的MCTS玩家
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout, is_selfplay=1)

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

    def collect_selfplay_data(self, n_games=1):
        """收集自我对抗数据用于训练"""
        for i in range(n_games):
            # 使用MCTS蒙特卡罗树搜索进行自我对抗
            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # 把翻转棋盘数据加到数据集里
            play_data = self.get_equi_data(play_data)
            # 保存对抗数据到data_buffer
            self.data_buffer.extend(play_data)

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
            loss, entropy = self.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch, self.learn_rate * self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:  # 如果D_KL跑偏则尽早停止
                break
        # 自动调整学习率
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 - np.var(np.array(winner_batch) - old_v.flatten()) / np.var(np.array(winner_batch)))
        explained_var_new = (1 - np.var(np.array(winner_batch) - new_v.flatten()) / np.var(np.array(winner_batch)))
        logger.info(("TRAIN kl:{:.5f},lr_multiplier:{:.3f},loss:{},entropy:{},explained_var_old:{:.3f},explained_var_new:{:.3f}"
                     ).format(kl, self.lr_multiplier, loss, entropy, explained_var_old, explained_var_new), GomokuTrainPipeline)
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """
        策略胜率评估：模型与纯MCTS玩家对战n局看胜率
        """
        # AlphaGo Zero风格的MCTS玩家（使用策略价值网络来指导树搜索和评估叶节点）
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)
        # 纯MCTS玩家
        pure_mcts_player = MCTSPurePlayer(c_puct=5, n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):  # 对战
            winner = self.game.start_play(current_mcts_player, pure_mcts_player, start_player=i % 2, is_shown=0)
            win_cnt[winner] += 1
        # 胜率
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        logger.info("TRAIN Num_playouts:{}, win: {}, lose: {}, tie:{}".format(self.pure_mcts_playout_num,
                                                                              win_cnt[1], win_cnt[2], win_cnt[-1]), GomokuTrainPipeline)
        return win_ratio

    def run(self):
        """启动训练"""
        try:
            for i in range(self.game_batch_num):  # 计划训练批次
                # 收集自我对抗数据
                self.collect_selfplay_data(self.play_batch_size)
                logger.info("TRAIN Batch i:{}, episode_len:{}".format(i + 1, self.episode_len), GomokuTrainPipeline)
                # 使用对抗数据重新训练策略价值网络模型
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                # 每n个batch检查一下当前模型胜率
                if (i + 1) % self.check_freq == 0:
                    logger.info("TRAIN Current self-play batch: {}".format(i + 1), GomokuTrainPipeline)
                    # 策略胜率评估：模型与纯MCTS玩家对战n局看胜率
                    win_ratio = self.policy_evaluate(self.policy_evaluate_size)
                    self.policy_value_net.save_model(CUR_PATH + '/model/current_policy_{}x{}.model'.format(self.board_width, self.board_height))
                    if win_ratio > self.best_win_ratio:  # 胜率超过历史最优模型
                        logger.info("TRAIN New best policy!!!!!!!!batch:{} win_ratio:{}->{} pure_mcts_playout_num:{}".format(i + 1, self.best_win_ratio, win_ratio, self.pure_mcts_playout_num),
                                    GomokuTrainPipeline)
                        self.best_win_ratio = win_ratio
                        # 保存当前模型为最优模型best_policy
                        self.policy_value_net.save_model(CUR_PATH + '/model/best_policy_{}x{}.model'.format(self.board_width, self.board_height))
                        # 如果胜率=100%，则增加纯MCT的模拟数 (<6000的限制视mem情况)
                        if (self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 6000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            logger.info('\n\rquit', GomokuTrainPipeline)


if __name__ == '__main__':
    #training_pipeline = GomokuTrainPipeline()
    model_file = CUR_PATH + '/model/current_policy_8x8.model'
    training_pipeline = GomokuTrainPipeline(init_model=model_file)
    training_pipeline.run()
