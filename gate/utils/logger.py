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
"""Logger to control display"""

import os
import time
import logging
from datetime import datetime
from gate.env import env


class Logger():
  """Logger helper"""

  def __init__(self):
    # init
    self.logger = logging.getLogger('TensorGate')
    self.logger.setLevel(logging.DEBUG)
    # control output
    self._DATE = env._LOG_DATE
    self._SYS = env._LOG_SYS
    self._TRAIN = env._LOG_TRAIN
    self._TEST = env._LOG_TEST
    self._VAL = env._LOG_VAL
    self._NET = env._LOG_NET
    self._WARN = env._LOG_WARN
    self._INFO = env._LOG_INFO
    self._ERR = env._LOG_ERR
    # config information
    self._CFG = env._LOG_CFG
    # control timer
    self._TIMER = env._LOG_TIMER
    self._start_timer = None

  def init(self, name, output_dir):
    """Initilize the logger"""
    logger_path = os.path.join(output_dir, name + '.log')
    self.set_filestream(logger_path)
    self.set_screenstream()

  def set_filestream(self, filepath, level=logging.DEBUG):
    """Setting content output to file"""
    fh = logging.FileHandler(filepath)
    fh.setLevel(level)
    self.logger.addHandler(fh)

  def set_screenstream(self, level=logging.DEBUG):
    """Setting content output to screen"""
    ch = logging.StreamHandler()
    ch.setLevel(level)
    self.logger.addHandler(ch)

  def _print(self, show_type, content):
    """Format print string"""
    if self._DATE:
      str_date = '[' + \
          datetime.strftime(datetime.now(), '%y.%m.%d %H:%M:%S') + '] '
      self.logger.info(str_date + show_type + ' ' + content)
    else:
      self.logger.info(show_type + ' ' + content)

  def sys(self, content):
    """Print information related to build system."""
    if self._SYS:
      self._print('[SYS]', content)

  def net(self, content):
    """Build net graph related infomation."""
    if self._NET:
      self._print('[NET]', content)

  def train(self, content):
    """Relate to the training processing."""
    if self._TRAIN:
      self._print('[TRN]', content)

  def val(self, content):
    """Relate to the validation processing."""
    if self._TRAIN:
      self._print('[VAL]', content)

  def test(self, content):
    """Relate to the test processing."""
    if self._TEST:
      self._print('[TST]', content)

  def warn(self, content):
    """Some suggest means warning."""
    if self._WARN:
      self._print('[WAN]', content)

  def info(self, content):
    """Just print it for check information"""
    if self._INFO:
      self._print('[INF]', content)

  def cfg(self, content):
    """Just print it for check information"""
    if self._CFG:
      self._print('[CFG]', content)

  def error(self, content):
    """For error info"""
    if self._ERR:
      self._print('[ERR]', content)

  def iters(self, cur_iter, keys, values):
    _data = 'Iter:%d' % cur_iter
    for i in range(len(keys)):
      if type(values[i]) == int or type(values[i]) == str:
        _data += ', %s:%s' % (str(keys[i]), str(values[i]))
      elif keys[i] == 'lr':
        _data += ', %s:%.6f' % (str(keys[i]), float(values[i]))
      else:
        _data += ', %s:%.4f' % (str(keys[i]), float(values[i]))
    return _data

  def start_timer(self):
    self._start_timer = time.time()

  def end_timer(self, content=''):
    if self._start_timer is None:
      raise ValueError('Call start_timer function first.')
    spend = time.time() - self._start_timer
    if self._TIMER:
      content = content + 'elapsed time:%.3fs' % spend
      self._print('[TIM]', content)
    self._start_timer = None


logger = Logger()
