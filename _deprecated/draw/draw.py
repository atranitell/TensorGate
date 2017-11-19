# -*- coding: utf-8 -*-
""" offer a set of draw curve method.
    Updated: 2017/06/11
"""
import argparse
import utils
import draw_line


def interface(args):
    """ a command manager tool
    """
    config = utils.json_parser(args.file)

    if config['figure']['type'] == 'basic_line_chart':
        draw_line.basic_line_chart(config)
    else:
        raise ValueError('Unknown input type.')


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('-file', type=str, default=None,
                        dest='file', help='path to model folder.')
    ARGS, _ = PARSER.parse_known_args()
    interface(ARGS)
