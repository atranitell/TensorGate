# -*- coding: utf-8
""" Logger to control display
"""
import os
import logging
from datetime import datetime
from core.env import env


class Logger():
  """ logger helper
  """

  def __init__(self):
    self.logger = logging.getLogger('TensorGate')
    self.logger.setLevel(logging.DEBUG)
    self._DATE = env._LOG_DATE
    self._SYS = env._LOG_SYS
    self._TRAIN = env._LOG_TRAIN
    self._TEST = env._LOG_TEST
    self._VAL = env._LOG_VAL
    self._NET = env._LOG_NET
    self._WARN = env._LOG_WARN
    self._INFO = env._LOG_INFO
    self._ERR = env._LOG_ERR

  def init(self, name, output_dir):
    """ initilize the logger
    """
    logger_path = os.path.join(output_dir, name + '.log')
    self.set_filestream(logger_path)
    self.set_screenstream()

  def set_filestream(self, filepath, level=logging.DEBUG):
    """ setting content output to file
    """
    fh = logging.FileHandler(filepath)
    fh.setLevel(level)
    self.logger.addHandler(fh)

  def set_screenstream(self, level=logging.DEBUG):
    """ setting content output to screen
    """
    ch = logging.StreamHandler()
    ch.setLevel(level)
    self.logger.addHandler(ch)

  def _print(self, show_type, content):
    """ format print string
    """
    if self._DATE:
      str_date = '[' + \
          datetime.strftime(datetime.now(), '%y.%m.%d %H:%M:%S') + '] '
      self.logger.info(str_date + show_type + ' ' + content)
    else:
      self.logger.info(show_type + ' ' + content)

  def sys(self, content):
    """ Print information related to build system.
    """
    if self._SYS:
      self._print('[SYS]', content)

  def net(self, content):
    """ build net graph related infomation.
    """
    if self._NET:
      self._print('[NET]', content)

  def train(self, content):
    """ relate to the training processing.
    """
    if self._TRAIN:
      self._print('[TRN]', content)

  def val(self, content):
    """ relate to the validation processing.
    """
    if self._TRAIN:
      self._print('[VAL]', content)

  def test(self, content):
    """ relate to the test processing.
    """
    if self._TEST:
      self._print('[TST]', content)

  def warn(self, content):
    """ some suggest means warning.
    """
    if self._WARN:
      self._print('[WAN]', content)

  def info(self, content):
    """ just print it for check information
    """
    if self._INFO:
      self._print('[INF]', content)

  def error(self, content):
    """ For error info
    """
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


logger = Logger()
