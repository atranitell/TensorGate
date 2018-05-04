# -*- coding: utf-8 -*-
"""
GATE FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2017/3/28

--------------------------------------------------------

Data UTILS

"""

import tensorflow as tf


def convert_to_tensor(res_list, type_list):
  """ convert python in-built type to tensor type
  """
  tensors = []
  for i, t in enumerate(type_list):
    if t == str:
      r = tf.convert_to_tensor(res_list[i], dtype=tf.string)
    elif t == int:
      r = tf.convert_to_tensor(res_list[i], dtype=tf.int32)
    elif t == float:
      r = tf.convert_to_tensor(res_list[i], dtype=tf.float32)
    else:
      raise ValueError('Unknown Input Type [%s]' % t)
    tensors.append(r)
  return tensors
