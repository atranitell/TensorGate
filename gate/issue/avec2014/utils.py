# -*- coding: utf-8 -*-
"""
GATE FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2018/4/26

--------------------------------------------------------

AVEC2014 UTILS for computing accuracy

Input: a log file
Output: {'class1':[label, logit], ...}

"""

import re
import math
import numpy as np


def get_list(path):
  """ For image input
      class1 2 3.56
      class1 2 4.32
      class1 2 1.23
      ...
      class2 3 2.11
      ...
  Return:
      Note: the logit will be mean of the value,
        because the label is same value in same class
      {'class1':[label, logit], ...}
  """
  res_fp = open(path, 'r')
  res_label = {}
  res_logit = {}
  for line in res_fp:
    line = line.replace('/', '\\')
    r1 = re.findall('frames_flow\\\(.*?)_video', line)
    r2 = re.findall('frames\\\(.*?)_video', line)

    res = r1[0] if len(r1) else r2[0]

    label = re.findall(' (.*?) ', line)
    logit = re.findall(label[0] + ' (.*)\n', line)

    if res not in res_label:
      res_label[res] = [float(label[0])]
    else:
      res_label[res].append(float(label[0]))

    logit_f = float(logit[0])

    if res not in res_logit:
      res_logit[res] = [logit_f]
    else:
      res_logit[res].append(logit_f)

  # acquire mean
  result = {}
  for idx in res_label:
    label = np.mean(np.array(res_label[idx]))
    logit = np.mean(np.array(res_logit[idx]))
    result[idx] = [label, logit]
  return result


def get_mae_rmse(res):
  """
  Input: a dict
    { 'class1':[label value],
      'class2':[label value] }
  """
  mae = 0.0
  rmse = 0.0
  for idx in res:
    mae += abs(res[idx][0] - res[idx][1])
    rmse += math.pow(res[idx][0] - res[idx][1], 2)
  mae = mae / len(res)
  rmse = math.sqrt(rmse / len(res))
  return mae, rmse, len(res)


def get_accurate_from_file(path):
  res = get_mae_rmse(get_list(path))
  mae, rmse, _ = res
  return mae, rmse
