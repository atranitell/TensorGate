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
"""Assemble model parametere and output to numpy file
The storgae method collect tensor name, and run in sesion, then collect all 
results in array.

1) add_tensor(tensor, attr, assigned_name)
2) result = session.run(run_tensor)
3) add(result)
4) dump

"""

import os
import pickle
import numpy as np
from gate.utils.logger import logger


class Storage():

  def __init__(self):
    """Any model include three parts"""
    self._tensor = {}
    self._idx_to_tensor_name = {}
    self._weight_unlock = True

  def add_tensor(self, tensor, attr, assigned_name=None):
    """attr: ['output', 'weight', 'source']
    """
    if assigned_name is not None:
      name = assigned_name
    else:
      name = tensor.name
    self._tensor[name] = {
        'tensor': tensor,
        'attr': attr,
        'shape': None,
        'data': []
    }

  @property
  def run_tensor(self):
    tensors = []
    for idx, name in enumerate(self._tensor):
      self._idx_to_tensor_name[idx] = name
      tensors.append(self._tensor[name]['tensor'])
    return tensors

  def add(self, numpy_list):
    """The return values from session
      source and output add for each batch
      weight add once only
    """
    for idx, value in enumerate(numpy_list):
      name = self._idx_to_tensor_name[idx]
      if self._tensor[name]['attr'] == 'weight':
        if self._weight_unlock:
          self._tensor[name]['data'] = value  # becasuse store once
      else:
        self._tensor[name]['data'].append(value)
    self._weight_unlock = False

  def dump(self, dst_name='model', dst_dir='./'):
    for name in self._tensor:
      self._tensor[name]['tensor'] = None  # pthread lock
      self._tensor[name]['data'] = np.array(self._tensor[name]['data'])
      self._tensor[name]['shape'] = self._tensor[name]['data'].shape
      logger.info("%s %s %s" % (name, self._tensor[name]['shape'],
                                self._tensor[name]['attr']))
    fpath = os.path.join(dst_dir, dst_name)
    self.dump_file(self._tensor, fpath+'.pkl')

  def dump_file(self, data, path):
    with open(path, 'wb') as fw:
      pickle.dump(data, fw)
