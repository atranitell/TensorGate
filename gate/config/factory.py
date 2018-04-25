# -*- coding: utf-8 -*-
"""
GATE FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2018/4/25

--------------------------------------------------------

Choose a task to execuate.

"""

from gate.config import mnist

config_map = {
    'mnist': mnist.MNIST
}


def get_config(dataset, config=None):
  """ dataset config factory
  Args:
    dataset: the name of specific task
    config: the config file for re-write dataset default config
  """
  return config_map[dataset](config)
