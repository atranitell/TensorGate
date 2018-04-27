# -*- coding: utf-8 -*-
""" offer a set of draw curve method.
    Updated: 2017/12/06
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import _parser


def draw_roc(config):
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
    _res = _parser.step(dt)
    roc = parse_roc(_res)
    FPR, TPR = zip(*roc)
    auc = compute_auc(FPR, TPR)
    print(dt['path'] + ' AUC: %f' % auc)
    plt.plot(FPR, TPR, label=fill('legend', None, dt), alpha=0.8)

  # additional line
  plt.plot([0, 1], [0, 1], 'k--', alpha=0.2)
  plt.plot([0, 1], [1, 0], 'k--', alpha=0.2)

  # step2: config figure
  loc = plticker.MultipleLocator(base=0.1)
  ax.xaxis.set_major_locator(loc)
  ax.yaxis.set_major_locator(loc)
  ax.grid(which='major', axis='both', linestyle='-')

  # label
  plt.title(fill('title', 'Line chart'))
  plt.xlabel(fill('xlabel', 'False Positive Rate'))
  plt.ylabel(fill('ylabel', 'True Positive Rate'))

  # lim
  plt.xlim(xmin=fill('xmin', 0))
  plt.xlim(xmax=fill('xmax', 1))
  plt.ylim(ymin=fill('ymin', 0))
  plt.ylim(ymax=fill('ymax', 1))

  # show legend
  plt.legend(loc=1)

  # save
  if 'save_fig' in cfg_fig:
    plt.savefig(cfg_fig['save_fig'])

  # plt show
  plt.show()


def parse_roc(result):
  """
  """
  size = len(result)
  roc_result = []
  for i in range(size):
    res = compute_roc(result, i)
    roc_result.append(res)
  roc_result.sort()
  return roc_result


def compute_roc(result, n_positive):
  """ result should be sorted first
  """
  result.sort()
  TP, FP, FN, TN = 0., 0., 0., 0.
  for idx, item in enumerate(result):
    if idx < n_positive:
      # prediction real
      if item[1] == 1:
        TP += 1.
      elif item[1] == -1 or item[1] == 0:
        FP += 1.
    else:
      # prediction false
      if item[1] == 1:
        FN += 1.
      elif item[1] == -1 or item[1] == 0:
        TN += 1.
  FPR = FP / (FP + TN)
  TPR = TP / (TP + FN)
  return FPR, TPR


def compute_auc(FPR, TPR):
  """
  """
  auc = 0.
  prev_x = 0
  for item in zip(FPR, TPR):
    if item[0] != prev_x:
      auc += (item[0] - prev_x) * item[1]
      prev_x = item[0]
  return auc
