# -*- coding: utf-8 -*-
""" offer a set of draw curve method.
    Updated: 2017/12/06
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import _parser


def draw_trend(config):
  """ 
  """
  # initail param
  plt.close('all')
  fig, ax = plt.subplots()

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
    x, v1, v2 = parse_trend(_parser.step(dt))
    plt.plot(x, v1, '.', label=fill('legend', 'label', dt), alpha=0.8)
    plt.plot(x, v2, '.', label=fill('legend', 'pred', dt), alpha=0.8)

  # step2: config figure
  plt.grid()

  # label
  plt.title(fill('title', 'Trend Demo'))
  plt.xlabel(fill('xlabel', 'index'))
  plt.ylabel(fill('ylabel', 'value'))

  # lim
  plt.xlim(xmin=fill('xmin', 0))
  plt.xlim(xmax=fill('xmax', None))
  plt.ylim(ymin=fill('ymin', None))
  plt.ylim(ymax=fill('ymax', None))

  # show legend
  plt.legend(loc=1)

  # save
  if 'save_fig' in cfg_fig:
    plt.savefig(cfg_fig['save_fig'])

  # plt show
  plt.show()


def parse_trend(result):
  """
  """
  result.sort()
  x = range(0, len(result))  
  v1 = []
  v2 = []
  for i in result:
    v1.append(i[0])
    v2.append(i[1])
  return x, v1, v2