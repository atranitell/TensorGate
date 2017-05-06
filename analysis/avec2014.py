# -*- coding: utf-8 -*-
""" Independently for avec2014 module
"""
import argparse
import avec2014_draw
import avec2014_ensemble


def interface(config):

    if config.task == 'draw_single_train':
        info = avec2014_draw.parse_log_train(config.file)
        avec2014_draw.draw_single(info, (0, 14))

    if config.task == 'draw_single_test':
        info = avec2014_draw.parse_log_test(config.file)
        avec2014_draw.draw_single(info, (7, 14))

    if config.task == 'draw_more':
        avec2014_draw.draw_multiple(config.file)

    if config.task == 'ensemble':
        avec2014_ensemble.get_ensemble_value(config.file)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '-task', type=str, default='draw_more', dest='task', help='')
    PARSER.add_argument(
        '-file', type=str, default=None, dest='file', help='')
    ARGS, _ = PARSER.parse_known_args()
    interface(ARGS)
