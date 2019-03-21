#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: game.py
Desc: 五子棋棋盘&对局
Author:yanjingang(yanjingang@mail.com)
Date: 2019/1/4 21:51
"""

from __future__ import print_function
import numpy as np


class Board(object):
    """棋盘"""

    def __init__(self, **kwargs):
        # 棋盘大小
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        # 棋盘状态（key: 棋盘位置, value: 落子玩家）
        self.states = {}
        # 多少个子连在一起才算赢
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2

    def init_board(self, start_player=0):
        """初始化棋盘"""
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width/height can not be less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        # 可落子位置
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        """
        位置类型转换(move->location)
        3*3棋盘move对应位置:
        6 7 8
        3 4 5
        0 1 2
        move=5时location=(1,2)
        """
        h = int(move // self.width)
        w = int(move % self.width)
        return [h, w]

    def location_to_move(self, location):
        """
        位置类型转换(location->move)
        3*3棋盘move对应位置:
        6 7 8
        3 4 5
        0 1 2
        location=(1,2)时move=5
        """
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """返回当前玩家角度的棋盘方格状态。形状：4*width*height"""
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]  # 当前玩家的落子位置
            move_oppo = moves[players != self.current_player]  # 对家的落子情况
            # 当前玩家棋子在棋盘方格中的状态
            square_state[0][move_curr // self.width, move_curr % self.height] = 1.0
            # 对家棋子在棋盘方格中的状态
            square_state[1][move_oppo // self.width, move_oppo % self.height] = 1.0
            # 最后一次落子在棋盘方格中的状态
            square_state[2][self.last_move // self.width, self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # 要播放的颜色
        return square_state[:, ::-1, :]  # 翻转对家和最后一次落子位置的棋盘视角

    def do_move(self, move):
        """落子"""
        # 保存棋盘状态
        self.states[move] = self.current_player
        # 更新可落子位置
        self.availables.remove(move)
        # 交替当前玩家
        self.current_player = (self.players[0] if self.current_player == self.players[1] else self.players[1])
        # 记录末次落子位置
        self.last_move = move

    def has_a_winner(self):
        """判断盘面是否有人胜出"""
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        # 有落子的位置
        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row + 2:
            return False, -1

        # 遍历落子位置，检查是否出现横/竖/斜线上n子相连的情况
        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """检查游戏是否结束"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):  # 棋盘无可落子位置，和棋
            return True, -1
        return False, -1

    def get_current_player(self):
        """返回当前执子玩家"""
        return self.current_player


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
            # 交替棋手
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
            move, move_probs = player.get_action(self.board, temp=temp, return_prob=1)
            # 保存当前盘面
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # 执行落子
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            # 检查游戏是否结束
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
