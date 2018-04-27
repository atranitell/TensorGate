# -*- coding: utf-8 -*-
"""
DECTRION FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2018/4/21

--------------------------------------------------------

FOR KINFACE SERIES

"""

from gate.issue.kinface.kinface_1E import KINFACE_1E

def select(config):
  """ select different subtask
  """
  if config.target == 'kinface.1E':
    return KINFACE_1E(config)
  elif config.target == 'kinface.2E':
    return
  elif config.target == 'kinface.1P1G':
    return
  elif config.target == 'kinface.1P1G1D':
    return
  elif config.target == 'kinface.1P1G.parallel':
    return
  else:
    raise ValueError('Unknown Target [%s]' % config.target)
