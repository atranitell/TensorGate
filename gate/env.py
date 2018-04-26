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

from gate.util import filesystem


class Env():
  """ Environment Variables.
  """

  def __init__(self):
    # setting output file
    self._OUTPUT = filesystem.mkdir('../_outputs/')
    self._DATASET = '../_datasets'

    # logger config
    self._LOG_DATE = True
    self._LOG_SYS = True
    self._LOG_TRAIN = True
    self._LOG_TEST = True
    self._LOG_VAL = True
    self._LOG_NET = True
    self._LOG_WARN = True
    self._LOG_INFO = True
    self._LOG_ERR = True
    self._LOG_CFG = True
    self._LOG_TIMER = True

    # compile config - processing *.py files
    self._COMPILE_DIR_BIN = '../_bin'  # output dir of binary file
    self._COMPILE_DIR_SRC = '../_src'  # output dir of code source
    self._COMPILE_EXCLUDE_ROOT = ['_', '.git', '.vscode']  # skip folders
    self._COMPILE_EXCLUDE_FILE = ['compile.py']  # skip files

    # SUMMARY SCALAR
    self._SUMMARY_SCALAR = True

    # SUMMARY SETTING
    self._SUMMARY_GRAD_STAT = False
    self._SUMMARY_GRAD_HIST = False
    self._SUMMARY_WEIGHT_STAT = False
    self._SUMMARY_WEIGHT_HIST = False


env = Env()
