# Copyright 2017 The KaiJIN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""UTILS"""

import os
import json
import math


def load_json(filepath):
  """Parse json file to dict"""
  if not os.path.isfile(filepath):
    raise ValueError('File could not find in %s' % filepath)
  with open(filepath, 'r') as fp:
    config = json.load(fp)
  return config


def save_json(dicts, filepath):
  """Save config to json file"""
  with open(filepath, 'w') as fp:
    json.dump(dicts, fp)


def downsampling(data, interval):
  """Downsampling data with interval.
  Args:
    data: a list.
    interval: a int type number.

  Returns:
    a new list with downsampling
  """
  length = len(data)
  assert interval > 0
  ret = []
  for idx in range(0, length, interval):
    ret.append(data[idx])
  return ret


def smooth(data, num):
  """K means"""
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

def clip(data, start_idx):
  """Return data[start_idx:]"""
  return data[start_idx:]

def process_keys(func, *args, ignore_keys=None):
  """For each key:

  Args:
    func: a function name
    args: args[0] should be dict, and the rest of is parameters
  """
  ret = {}
  for var in args[0]:
    if ignore_keys is None or var not in ignore_keys:
      ret[var] = func(args[0][var], *args[1:])
  return ret
