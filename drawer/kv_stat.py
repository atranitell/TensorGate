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
"""Key-Value for the event file from TensorGate"""

import os
import numpy as np
from drawer import utils
from drawer import data_parser


def compute_kv(config):
  """Parse log data and calling draw"""
  result = {}
  for _cfg in config['data']:
    data = data_parser.log_kv(_cfg['path'], _cfg['phase'], _cfg['keys'])

    # clip from start idx
    if 'start_iter' in _cfg:
      start_idx = 0
      for idx, iteration in enumerate(data['iter']):
        if iteration >= _cfg['start_iter']:
          start_idx = idx
          break
      data = utils.process_keys(utils.clip, data, start_idx)

    # downsampling all points including iter
    if 'iter_invl' in _cfg:
      invl = int(_cfg['iter_invl'] / (data['iter'][1]-data['iter'][0]))
      assert invl >= 1
      data = utils.process_keys(utils.downsampling, data, invl)

    res_list = {}
    # compute max
    if _cfg['task'] == 'max':
      idx, value = _kv_max(data, _cfg['sort_key'])
      # broadcast to other key
      res_list['iter'] = data['iter'][idx]
      for key in _cfg['keys']:
        res_list[key] = data[key][idx]
    
    elif _cfg['task'] == 'min':
      idx, value = _kv_min(data, _cfg['sort_key'])
      # broadcast to other key
      res_list['iter'] = data['iter'][idx]
      for key in _cfg['keys']:
        res_list[key] = data[key][idx]

    # print
    print(_cfg['path'])
    for res in res_list:
      print('  ', res, res_list[res])

    # add-in result
      result[os.path.basename(_cfg['path'])] = data

  return result


def compute_kv_template(config):
  """Draw from a line template"""
  # copy to a new memory, avoid lost info
  _i = config['info'].copy()
  _d = config['data'].copy()

  # fill in the data
  if len(_d) < 1:
    raise ValueError('A template data is essential!')

  config['data'] = []
  for log_name in os.listdir(_i['dir']):
    if log_name.split('.')[-1] != 'log':
      continue
    config['data'].append(_d[0].copy())
    config['data'][-1]['path'] = os.path.join(_i['dir'], log_name)

  # output to jsonfile
  output_filename = os.path.basename(_i['dir'])+'.'+_i['task'] + '.json'
  output = os.path.join(_i['dir'], output_filename)

  # change type
  config['task'] = _i['task']
  config.pop('info')

  utils.save_json(config, output)
  print('Config file has been saved in %s' % output)

  if _i['run']:
    return compute_kv(config)


def compute_tv_template_folder(config):
  """Traverse for folder"""
  root = config['info']['dir']
  results = {}
  for folder in os.listdir(root):
    sub_config = config.copy()
    sub_config['info']['dir'] = os.path.join(root, folder)
    results[folder] = compute_kv_template(sub_config)
    print('\n\n\n')
  utils.save_json(results, config['info']['save_to_json'])


def _kv_max(data, sort_key):
  """Compute maximum value for a line."""
  max_idx = np.argmax(data[sort_key])
  return max_idx, data[sort_key][max_idx]

def _kv_min(data, sort_key):
  """Compute minimum value for a line."""
  min_idx = np.argmin(data[sort_key])
  return min_idx, data[sort_key][min_idx]
