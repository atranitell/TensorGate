# -*- coding: utf-8 -*-
""" offer a set of draw curve method.
    Updated: 2017/06/11
"""
import argparse
import _line
import _roc
import _trend
import utils


def interface(args):
  """ a command manager tool """
  config = utils.parse_json(args.file)
  if config['figure']['type'] == 'basic_line_chart':
    _line.draw_basic_line_chart(config)
  elif config['figure']['type'] == 'roc':
    _roc.draw_roc(config)
  elif config['figure']['type'] == 'trend':
    _trend.draw_trend(config)
  else:
    raise ValueError('Unknown input type.')


if __name__ == "__main__":
  PARSER = argparse.ArgumentParser()
  PARSER.add_argument('-file', type=str, default=None,
                      dest='file', help='path to model folder.')
  ARGS, _ = PARSER.parse_known_args()
  interface(ARGS)
