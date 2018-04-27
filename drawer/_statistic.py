# -*- coding: utf-8 -*-
""" acquire a series of statistic value
    Updated: 2018/04/27
"""

import os
import utils
import _parser


def print_info(config, root):
  """
  """
  dir_path = os.path.join(root, config['info']['dir'])
  for dt in config['data']:
    for log in os.listdir(dir_path):
      if log.split('.')[-1] != 'log':
        continue
      dt['path'] = os.path.join(dir_path, log)
      res = _parser.bigram(dt['path'], dt['phase'], dt['key'])
      if dt['stat'] == 'max':
        _print_max(res, dt)
      elif dt['stat'] == 'min':
        _print_min(res, dt)


def _print_min(res, dt):
  val, step = utils.find_min_value(res[dt['key']], res['iter'], dt['invl'])
  _print_value(step, val, dt['path'], '%s[%s][min]' % (dt['phase'], dt['key']))


def _print_max(res, dt):
  val, step = utils.find_max_value(res[dt['key']], res['iter'], dt['invl'])
  _print_value(step, val, dt['path'], '%s[%s][max]' % (dt['phase'], dt['key']))


def _print_value(step, value, path, content):
  output = 'Path:%s, Iter:%d, Value:%f, Attr:%s'
  print(output % (os.path.basename(path), step, value, content))
