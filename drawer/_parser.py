# -*- coding: utf-8 -*-
""" offer a set of draw curve method.
    Updated: 2017/12/06
"""
import os
import re


def bigram(filepath, phase, key):
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
      if not r_key:
        r_key = re.findall(key + ':(.*).', line)
      if r_iter and r_key:
        data['iter'].append(int(r_iter[0]))
        data[key].append(float(r_key[0]))

  # check equal
  assert len(data['iter']) == len(data[key])
  return data


def step(config):
  """ parse step result
  """
  result = []
  with open(config['path']) as fp:
    for line in fp:
      res = line.split(' ')
      if not res:
        continue
      _entries = []
      for entry in config['idx']:
        _v = res[entry['pos'] - 1]
        _v = int(_v) if entry['int'] else float(_v)
        _entries.append(_v)
      result.append(_entries)
  return result
