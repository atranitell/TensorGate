# -*- coding: utf-8 -*-
""" offer a set of draw curve method.
    Updated: 2017/06/11
"""

import os
import argparse
import _statistic
import _line
import _roc
import _trend
import _config
import utils


def interface(args):
  """ a command manager tool """
  config = utils.parse_json(args.file)
  root = os.path.dirname(args.file)
  if config['figure']['type'] == 'basic_line_chart':
    _line.draw_basic_line_chart(config, root)
  elif config['figure']['type'] == 'basic_line_chart_template':
    _config.gen_line_config(config, root)
  elif config['figure']['type'] == 'roc':
    _roc.draw_roc(config)
  elif config['figure']['type'] == 'trend':
    _trend.draw_trend(config)
  elif config['figure']['type'] == 'stat':
    _statistic.print_info(config, root)
  else:
    raise ValueError('Unknown input type.')


if __name__ == "__main__":
  PARSER = argparse.ArgumentParser()
  PARSER.add_argument('-file', type=str, default=None, dest='file')
  ARGS, _ = PARSER.parse_known_args()
  interface(ARGS)
