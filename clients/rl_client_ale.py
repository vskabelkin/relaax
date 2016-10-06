from __future__ import print_function

from socketIO_client import SocketIO
import logging

from time import sleep

from rl_client_ale.game_process import GameProcess as Game
from rl_client_ale.params import Params

import server_api


class ServerAPI(server_api.ServerAPI):
    def __init__(self, *args, **kwargs):
        server_api.ServerAPI.__init__(self, Params(), *args, **kwargs)

    def make_game(self, seed):
        return Game(seed, self.cfg.game_rom)

    def make_display_game(self, seed):
        return Game(seed, self.cfg.game_rom, display=True, no_op_max=0)

    def action_size(self):
        return self.gameList[0].real_action_size()  

    def game_state(self, i):
        return self.gameList[i].s_t

    def act(self, i, action):
        self.gameList[i].process(action)
        return self.gameList[i].reward

    def stop_play_thread(self):
        self.play_thread.join()
        sleep(3)
        self.gameList.pop()


if __name__ == "__main__":
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.basicConfig(level=logging.INFO)

    socketIO = SocketIO('localhost', 8000)
    rlmodels_namespace = socketIO.define(ServerAPI, '/rlmodels')
    socketIO.wait(seconds=1)