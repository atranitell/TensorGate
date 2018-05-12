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
"""LOG PARSER for the event file from TensorGate"""

import os
import re


def log_kv(filepath: str, phase: str, keys: list):
  """All input will be not case-sensitive phase and key should be same line.

  Args:
    filepath: the path to log file.
    phase: [TRN], [TST], [VAL].
    keys: a list like ['loss', 'mae', 'rmse', 'error']

  Returns:
    data: a dict:
      data['iter']: a list <>
      data[key]: a list <>

  """
  if not os.path.exists(filepath):
    raise ValueError('File could not find in %s' % filepath)

  # transfer to lower case
  phase = phase.lower()

  # return data
  data = {}
  data['iter'] = []
  for key in keys:
    key = key.lower()
    data[key] = []

  # parse
  with open(filepath, 'r') as fp:
    for line in fp:
      line = line.lower()
      if line.find(phase) < 0:
        continue
      # record iteration
      r_iter = re.findall('iter:(.*?),', line)
      data['iter'].append(int(r_iter[0]))
      # find each matched key
      for key in keys:
        key = key.lower()
        r_key = re.findall(key + ':(.*?),', line)
        if not r_key:
          r_key = re.findall(key + ':(.*).', line)
        if r_iter and r_key:
          data[key].append(float(r_key[0]))

  # check equal
  for key in keys:
    assert len(data['iter']) == len(data[key])

  return data


# def step(config):
#   """Parse data file like:
#     iter value1 value2 ...

#   """
#   result = []
#   with open(config['path']) as fp:
#     for line in fp:
#       res = line.split(' ')
#       if not res:
#         continue
#       _entries = []
#       for entry in config['idx']:
#         _v = res[entry['pos'] - 1]
#         _v = int(_v) if entry['int'] else float(_v)
#         _entries.append(_v)
#       result.append(_entries)
#   return result
