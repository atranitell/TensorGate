# -*- coding: utf-8 -*-
"""
DECTRION FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2018/4/21

--------------------------------------------------------

FOR AVEC2014 SERIES

"""

from gate.issue.avec2014.avec2014_audio_cnn import AVEC2014_AUDIO_CNN
from gate.issue.avec2014.avec2014_img_cnn import AVEC2014_IMG_CNN
from gate.issue.avec2014.avec2014_img_4view import AVEC2014_IMG_4VIEW
from gate.issue.avec2014.avec2014_img_bicnn import AVEC2014_IMG_BICNN


def select(config):
  """ select different subtask
  """
  if config.target == 'avec2014.img.cnn':
    return AVEC2014_IMG_CNN(config)
  elif config.target == 'avec2014.img.4view':
    return AVEC2014_IMG_4VIEW(config)
  elif config.target.startswith('avec2014.img.bicnn'):
    return AVEC2014_IMG_BICNN(config)
  elif config.target == 'avec2014.audio.cnn':
    return AVEC2014_AUDIO_CNN(config)
  else:
    raise ValueError('Unknown Target [%s]' % config.target)
