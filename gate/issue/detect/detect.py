# -*- coding: utf-8 -*-
"""
DECTRION FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2018/4/21

--------------------------------------------------------

DETECT

"""

from gate.issue.detect.mask_rcnn import MASK_RCNN

def select(config):
  """ select different subtask
  """
  if config.target == 'detect.mask_rcnn':
    return MASK_RCNN(config)
  else:
    raise ValueError('Unknown Target [%s]' % config.target)
