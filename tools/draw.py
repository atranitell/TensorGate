# -*- coding: utf-8 -*-
""" offer a set of draw curve method.
    Updated: 2017/06/11
"""
import argparse
import os
import json
import math
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker


def draw_basic_line_chart(config):
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
    res = parse_bigram(dt['path'], dt['phase'], dt['key'])
    # downsampling
    if dt['invl'] > 1:
      res = downsampling_bigram(res, dt['invl'])
    # smooth curve
    if dt['smooth'] > 1:
      res[dt['key']] = smooth(res[dt['key']], dt['smooth'])
    plt.plot(res['iter'], res[dt['key']],
             label=fill('legend', None, dt), alpha=0.8)
    # save data
    if 'save_data' in dt:
      write_to_text(res, ['iter', dt['key']], dt['save'])

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


def draw_roc(config):
  """ 
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
    roc = parse_roc(dt['path'], dt['roc'])
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


def parse_roc(filepath, roc):
  """
  """
  result = []
  with open(filepath) as fp:
    for line in fp:
      label = int(line.split(' ')[roc['n_label']])
      pred = float(line.split(' ')[roc['n_pred']])
      result.append((pred, label))

  size = len(result)
  roc_result = []
  for i in range(size):
    res = compute_roc(result, i, roc)
    roc_result.append(res)
  roc_result.sort()
  return roc_result


def compute_roc(result, n_positive, roc):
  """ result should be sorted first
  """
  result.sort()
  TP, FP, FN, TN = 0., 0., 0., 0.
  for idx, item in enumerate(result):
    if idx < n_positive:
      # prediction real
      if item[1] == roc['positive_label']:
        TP += 1.
      elif item[1] == roc['negtive_label']:
        FP += 1.
    else:
      # prediction false
      if item[1] == roc['positive_label']:
        FN += 1.
      elif item[1] == roc['negtive_label']:
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


def parse_bigram(filepath, phase, key):
  """
  All input will be not case-sensitive
      phase and key should be same line.
      each line should has key iter

  Args:
      filepath: the path to log file.
      phase: [TRN], [TST], [VAL].
      key: like loss, mae, rmse, error
  """
  if not os.path.exists(filepath):
    raise ValueError('File could not find in %s' % filepath)

  # transfer to lower case
  phase = phase.lower()
  key = key.lower()

  # return data
  data = {}
  data['iter'] = []
  data[key] = []

  # parse
  with open(filepath, 'r') as fp:
    for line in fp:
      line = line.lower()
      if line.find(phase) < 0:
        continue
      r_iter = re.findall('iter:(.*?),', line)
      r_key = re.findall(key + ':(.*?),', line)
      if len(r_key) == 0:
        r_key = re.findall(key + ':(.*).', line)
      if len(r_iter) and len(r_key):
        data['iter'].append(int(r_iter[0]))
        data[key].append(float(r_key[0]))

  # check equal
  assert len(data['iter']) == len(data[key])
  return data


def downsampling(data, interval):
  """ data is a list.
      interval is a int type number.
      return a new list with downsampling
  """
  length = len(data)
  assert interval > 0
  ret = []
  for idx in range(0, length, interval):
    ret.append(data[idx])
  return ret


def downsampling_bigram(data, interval):
  """ a simple interface for data like:
      data['iter'] = [...]
      data[key] = [...]
      it will travser all key in data and to downsampling
  """
  ret = {}
  for var in data:
    ret[var] = downsampling(data[var], interval)
  return ret


def smooth(data, num):
  """ K means
  """
  ret = []
  mid = math.floor(num / 2.0)
  for i in range(len(data)):
    if i > mid and i < len(data) - num:
      avg = 0
      for j in range(num):
        avg = avg + data[i + j]
      ret.append(avg / num)
    else:
      ret.append(data[i])
  return ret


def write_to_text(data, keys, filepath):
  """ data should be a dict
      keys is a key in dict
  ps:
      all key should has number of element in data.
      len(data[key0]) == len(data[key1]) == ...
  """
  for key in keys:
    if key not in data:
      raise ValueError('%s not in data' % key)

  fw = open(filepath, 'w')
  for idx in range(len(data[keys[0]])):
    line = ''
    for key in keys:
      line += str(data[key][idx]) + ' '
    line += '\n'
    fw.write(line)
  fw.close()


def parse_json(filepath):
  """ parse json file to dict
  """
  if not os.path.isfile(filepath):
    raise ValueError('File could not find in %s' % filepath)
  with open(filepath, 'r') as fp:
    config = json.load(fp)
  return config


def interface(args):
  """ a command manager tool """
  config = parse_json(args.file)
  if config['figure']['type'] == 'basic_line_chart':
    draw_basic_line_chart(config)
  elif config['figure']['type'] == 'roc':
    draw_roc(config)
  else:
    raise ValueError('Unknown input type.')


if __name__ == "__main__":
  PARSER = argparse.ArgumentParser()
  PARSER.add_argument('-file', type=str, default=None,
                      dest='file', help='path to model folder.')
  ARGS, _ = PARSER.parse_known_args()
  interface(ARGS)
