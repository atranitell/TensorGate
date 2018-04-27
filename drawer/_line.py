# -*- coding: utf-8 -*-
""" offer a set of draw curve method.
    Updated: 2017/12/06
"""

import os
import utils
import matplotlib.pyplot as plt
import _parser


def draw_basic_line_chart(config, root):
  """ config is a dict including:
  1. legend
  2. path: path to log
  3. phase: '[TRN]', '[TST]', '[VAL]'
  4. type: 'loss', 'err', 'mae', etc.
  5. invl: default to 1
  6. y_min, y_max, x_min, x_max: default to None
  7. smooth: default to 1, average with multi-points.
  8. show: default to true, to draw on the figure.
  """
  # initail param
  plt.close('all')
  cfg_fig = config['figure']
  cfg_data = config['data']

  def fill(key, default, cfg=cfg_fig):
    return cfg[key] if key in cfg else default

  # step1: parse data
  for dt in cfg_data:
    # control showing on figure
    if dt['show'] is False:
      continue
    # parse data from file
    dt['path'] = os.path.join(root, dt['path'])
    res = _parser.bigram(dt['path'], dt['phase'], dt['key'])
    # downsampling
    if dt['invl'] > 1:
      res = utils.downsampling_bigram(res, dt['invl'])
    # smooth curve
    if dt['smooth'] > 1:
      res[dt['key']] = utils.smooth(res[dt['key']], dt['smooth'])

    # draw plot
    style = fill('style', '-', dt)
    label = fill('legend', None, dt)
    color = fill('color', None, dt)
    alpha = fill('alpha', 0.8, dt)

    plt.plot(res['iter'], res[dt['key']], style,
             color=color, label=label, alpha=alpha)

    # save data
    if 'save_data' in dt:
      _max_v, _max_i = utils.find_max_value(res[dt['key']], res['iter'])
      _min_v, _min_i = utils.find_min_value(res[dt['key']], res['iter'])
      print('file:%s, iter:%d, max:%f' % (dt['path'], _max_i, _max_v))
      print('file:%s, iter:%d, min:%f' % (dt['path'], _min_i, _min_v))
      save_path = dt['path'].split('.log')[0] + '_sum.txt'
      utils.write_to_text(res, ['iter', dt['key']], save_path)

  # step2: config figure
  plt.grid()

  # label
  plt.title(fill('title', 'Line chart'))
  plt.xlabel(fill('xlabel', 'iter'))
  plt.ylabel(fill('ylabel', 'value'))

  # lim
  plt.xlim(xmin=fill('xmin', 0))
  plt.xlim(xmax=fill('xmax', None))
  plt.ylim(ymin=fill('ymin', None))
  plt.ylim(ymax=fill('ymax', None))

  # show legend
  plt.legend()

  # save
  if 'save_fig' in cfg_fig:
    plt.savefig(cfg_fig['save_fig'])

  # plt show
  plt.show()
