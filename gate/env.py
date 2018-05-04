# -*- coding: utf-8 -*-
"""
DECTRION FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2018/4/19

--------------------------------------------------------

ENVIRONMENT FOR COMMON CALLING

"""


class Env():
  """ Environment Variables.
  """

  def __init__(self):
    # logger config
    self._LOG_DATE = True
    self._LOG_SYS = False
    self._LOG_TRAIN = True
    self._LOG_TEST = True
    self._LOG_VAL = True
    self._LOG_NET = True
    self._LOG_WARN = True
    self._LOG_INFO = True
    self._LOG_ERR = True
    self._LOG_CFG = True


env = Env()
