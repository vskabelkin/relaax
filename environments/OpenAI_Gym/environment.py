from __future__ import print_function

import logging
import os
import random
import time
import signal

from relaax.client import rlx_client

from . import game_process


def run(rlx_server, env, seed):
    n_game = 0
    game = game_process.GameProcessFactory(env).new_env(_seed(seed))

    def toggle_rendering():
        if game.display:
            game._close_display = True
        else:
            game.display = True
            game._close_display = False

    signal.signal(signal.SIGUSR1, lambda _1, _2: toggle_rendering())
    signal.siginterrupt(signal.SIGUSR1, False)

    while True:
        try:
            with rlx_client.Client(rlx_server) as client:
                action = client.init(game.state())
                while True:
                    reward, reset = game.act(action)
                    if reset:
                        episode_score = client.reset(reward)
                        n_game += 1
                        print('Score at game', n_game, '=', episode_score)
                        game.reset()
                        action = client.send(None, game.state())
                    else:
                        action = client.send(reward, game.state())
        except rlx_client.Failure as e:
            _warning('{} : {}'.format(rlx_server, e.message))
            delay = random.randint(1, 10)
            _info('waiting for %ds...', delay)
            time.sleep(delay)


def _seed(value):
    if value is None:
        return random.randrange(1000000)
    return value


def _info(message, *args):
    logging.info('%d:' + message, os.getpid(), *args)


def _warning(message, *args):
    logging.warning('%d:' + message, os.getpid(), *args)
