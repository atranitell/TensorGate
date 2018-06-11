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
    self.args = args
    if args.config is not None:
      filesystem.raise_path_not_exist(args.config)
      self.EXTRA_CONFIG = self._load_config_file(args.config)

  def rewrite_command_args(self):
    """command line is first priority"""
    if self.args.task is not None:
      if self.args.task not in TASK_MAP:
        raise ValueError('Unknown task %s' % self.args.task)
      self.task = self.args.task
    if self.args.model is not None:
      filesystem.raise_path_not_exist(self.args.model)
      self.output_dir = self.args.model

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

  def _read_config_file(self, default_v, key_v):
    """Parse entry from config json. If it could not pick the value from config

    Example:
      In config file:
        { "train.entry_path": "../_datasets/train.txt" }
      In config py:
        r('../_datasets/train.txt', 'train.entry_path')
    """
    if self.EXTRA_CONFIG is not None  \
            and key_v in self.EXTRA_CONFIG  \
            and self.EXTRA_CONFIG[key_v] is not None:
      return self.EXTRA_CONFIG[key_v]
    return default_v
