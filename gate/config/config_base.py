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
"""CONFIG BASE"""

import json
from gate.utils import filesystem
from gate.utils.logger import logger


TASK_MAP = ['train', 'test', 'val', 'inference', 'extract_feature',
            'heatmap', 'freeze_model']


class Configbase():
  """All config class should inherit this class."""

  def __init__(self, args):
    """Check setting and path."""
    self.EXTRA_CONFIG = None
    if args.task is not None:
      if args.task not in TASK_MAP:
        raise ValueError('Unknown task %s' % args.task)
      self.task = args.task
    if args.model is not None:
      filesystem.raise_path_not_exist(args.model)
      self.output_dir = args.model
    if args.config is not None:
      filesystem.raise_path_not_exist(args.config)
      self.EXTRA_CONFIG = self._load_config_file(args.config)

  def set_phase(self, phase):
    """Switch system phase"""
    self.phase = phase
    if phase == 'train':
      self._train()
    elif phase == 'test':
      self._test()
    elif phase == 'val':
      self._val()
    elif phase == 'inference':
      self._inference()
    elif phase == 'heatmap':
      self._heatmap()
    else:
      raise ValueError('Unknown phase %s' % phase)

  def _train(self):
    self.data = self.train.data

  def _test(self):
    self.data = self.test.data

  def _val(self):
    self.data = self.val.data

  def _inference(self):
    self.data = self.inference.data

  def _heatmap(self):
    self.data = self.heatmap.data

  @staticmethod
  def _load_config_file(config_path):
    with open(config_path) as fp:
      return json.load(fp)

  def r(self, default_v, key_v):
    """key_v like 'train.lr' -> [train][lr]

    Example:
      r('../_datasets/train', 'data_dir')
      if config['data_dir'] has value, return config['data_dir']
        else return '../_datasets/train'

    """
    if self.EXTRA_CONFIG is None:
      raise ValueError('System has not loaded the extra config file.')
    r = key_v.split('.')
    try:
      if len(r) == 1:
        config_v = self.EXTRA_CONFIG[r[0]]
      elif len(r) == 2:
        config_v = self.EXTRA_CONFIG[r[0]][r[1]]
      elif len(r) == 3:
        config_v = self.EXTRA_CONFIG[r[0]][r[1]][r[2]]
      elif len(r) == 4:
        config_v = self.EXTRA_CONFIG[r[0]][r[1]][r[2]][r[3]]
      else:
        raise ValueError('Too long to implement!')
      v = config_v
    except BaseException:
      content = 'Key {0:,} has not found, using default {1:,}'
      logger.warn(content.format(key_v, default_v))
      v = default_v
    return v
