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
from gate.config import trafficflow
from gate.config import avec2014
from gate.config import kinface

config_map = {
    'mnist': mnist.MNIST,
    'trafficflow': trafficflow.TRAFFICFLOW,
    'avec2014': avec2014.AVEC2014,
    'avec2014.heatmap': avec2014.AVEC2014_HEATMAP,
    'avec2014.flow': avec2014.AVEC2014_FLOW,
    'avec2014.bicnn': avec2014.AVEC2014_BICNN,
    'avec2014.audio': avec2014.AVEC2014_AUDIO,
    'kinface.vae': kinface.KinfaceVAE
}


def get_config(dataset, config=None):
  """ dataset config factory
  Args:
    dataset: the name of specific task
    config: the config file for re-write dataset default config
  """
  return config_map[dataset](config)
