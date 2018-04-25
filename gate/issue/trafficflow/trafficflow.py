# -*- coding: utf-8 -*-
"""
DECTRION FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2018/4/21

--------------------------------------------------------

FOR TrafficNet

"""

from gate.issue.trafficflow.vanilla_cnn import VanillaCNN


def select(config):
  """ select different subtask
  """
  if config.target == 'trafficflow.vanilla':
    return VanillaCNN(config)
  else:
    raise ValueError('Unknown Target [%s]' % config.target)
