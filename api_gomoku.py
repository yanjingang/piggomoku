#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: api_gomoku.py
Desc: 五子棋 强化学习模型 API 封装
Demo: 
    cd /home/work/piglab/webservice/service/ && nohup python api_gomoku.py > log/api_gomoku.log &
    
    http://www.yanjingang.com:8023/piglab/game/gomoku?session_id=1548849426270&location=3,4

    ps aux | grep api_gomoku.py |grep -v grep| cut -c 9-15 | xargs kill -9
Author: yanjingang(yanjingang@mail.com)
Date: 2019/1/29 23:08
"""

import sys
import os
import json
import time
import logging
import numpy as np
import tornado.ioloop
import tornado.web
import tornado.httpserver

# PATH
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.realpath(CUR_PATH + '/../../')
sys.path.append(BASE_PATH)
# print(CUR_PATH, BASE_PATH)
from machinelearning.lib import utils
from game import Board, Game
from mcts import MCTSPlayer
from net.policy_value_net_keras import PolicyValueNet


class ApiGameGomoku(tornado.web.RequestHandler):
    """API逻辑封装"""
    width, height = 8, 8
    model_file = CUR_PATH + '/model/best_policy_8x8.model'
    best_policy = PolicyValueNet(width, height, model_file=model_file)
    games = {}

    def get(self):
        """get请求处理"""
        try:
            result = self.execute()
        except:
            logging.error('execute fail ' + utils.get_trace())
            result = {'code': 1, 'msg': '请求失败'}
        logging.info('API RES[' + self.request.path + '][' + self.request.method + ']['
                     + str(result['code']) + '][' + str(result['msg']) + '][' + str(result['data']) + ']')
        self.write(json.dumps(result))

    def post(self):
        """post请求处理"""
        try:
            result = self.execute()
        except:
            logging.error('execute fail ' + utils.get_trace())
            result = {'code': 1, 'msg': '请求失败'}
        logging.info('API RES[' + self.request.path + '][' + self.request.method + ']['
                     + str(result['code']) + '][' + str(result['msg']) + ']')
        self.write(json.dumps(result))

    def execute(self):
        """执行业务逻辑"""
        logging.info('API REQUEST INFO[' + self.request.path + '][' + self.request.method + ']['
                     + self.request.remote_ip + '][' + str(self.request.arguments) + ']')
        session_id = self.get_argument('session_id', '')
        res = {'player':-1, 'location': [], 'end': False, 'winner': -1,'session_id':session_id,'curr_player':0}
        location = self.get_argument('location', '').split(',')
        if len(location) == 2:
            location = [int(n) for n in location]
        if session_id == '':
            return {'code': 2, 'msg': 'session_id不能为空', 'data': res}

        try:
            # 1.新的对局
            session = {}
            if session_id not in self.games:
                logging.info("[{}] init new game!".format(session_id))
                # 初始化棋盘
                board = Board(width=self.width, height=self.height, n_in_row=5)
                board.init_board()
                session['game'] = Game(board)
                # 初始化AI棋手
                session['ai_player'] = MCTSPlayer(self.best_policy.policy_value_fn, c_puct=5, n_playout=500)
                self.games[session_id] = session
            else:
                session = self.games[session_id]
                #clear old games
                for k in list(self.games.keys()):
                    if int(time.time())-int(k)/1000 > 60*40: #超过40分钟的session清理
                        del(self.games[k])
                        logging.warning("[{}] timeout clear!".format(k))
            # 2.get ai move
            res['curr_player'] = session['game'].board.get_current_player() -1
            res['availables'] = session['game'].board.availables
            if res['curr_player'] == 0:  #ai固定执白(轮到ai时，忽略传入的move参数)
                location = res['location'] = session['game'].board.move_to_location(session['ai_player'].get_action(session['game'].board))
                logging.info("[{}] {} AI move: {}".format(session_id, res['curr_player'], location))
            else:
                logging.info("[{}] {} Human move: {}".format(session_id, res['curr_player'], location))

            # 3.do move
            if len(location) < 2 and res['curr_player']==1:
                return {'code': 2, 'msg': 'location不能为空', 'data': res}
            if len(location) == 2:
                #human or ai move
                move =  session['game'].board.location_to_move(location)
                session['game'].board.do_move(move)
                res['player'] = res['curr_player']
                res['end'], res['winner'] = session['game'].board.game_end()
                res['winner'] -= 1
                res['curr_player'] = session['game'].board.get_current_player() -1
                res['availables'] = session['game'].board.availables
                if res['end']:
                    return {'code': 0, 'msg': 'success', 'data': res}
                #ai move
                if res['curr_player'] == 0:  # ai固定执白
                    location = res['location'] = session['game'].board.move_to_location(session['ai_player'].get_action(session['game'].board))
                    logging.info("[{}] {} AI move: {}".format(session_id, res['curr_player'], location))
                    move =  session['game'].board.location_to_move(location)
                    session['game'].board.do_move(move)
                    res['player'] = res['curr_player']
                    res['end'], res['winner'] = session['game'].board.game_end()
                    res['winner'] -= 1
                    res['curr_player'] = session['game'].board.get_current_player() -1
                    res['availables'] = session['game'].board.availables
                    return {'code': 0, 'msg': 'success', 'data': res}

        except:
            logging.error('execute fail [' + str(location) + '][' + session_id + '] ' + utils.get_trace())
            return {'code': 5, 'msg': '请求失败', 'data': res}

        # 组织返回格式
        return {'code': 0, 'msg': 'success', 'data': res}

if __name__ == '__main__':
    """服务入口"""
    port = 8023

    # log init
    log_file = ApiGameGomoku.__name__.lower() # + '-' + str(os.getpid())
    utils.init_logging(log_file=log_file, log_path=CUR_PATH)
    print("log_file: {}".format(log_file))

    # 路由
    app = tornado.web.Application(
        handlers=[
            (r'/piglab/game/gomoku', ApiGameGomoku)
        ]
    )

    # 启动服务
    http_server = tornado.httpserver.HTTPServer(app, xheaders=True)
    http_server.listen(port)
    tornado.ioloop.IOLoop.instance().start()
