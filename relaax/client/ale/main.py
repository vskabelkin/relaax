from __future__ import print_function

import argparse
import logging
import os

from .client import run


def main():
    logging.basicConfig(
        format='%(asctime)s:%(levelname)s: %(message)s',
        level=logging.INFO
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--rlx-server', type=str, default=None, help='RLX server address (host:port)')
    parser.add_argument('--rom', type=str, help='Atari game ROM file')
    parser.add_argument('--seed', type=int, default=None, help='Seed for random generator')
    args = parser.parse_args()

    run(
        rlx_server=args.rlx_server,
        rom=args.rom,
        seed=args.seed
    )