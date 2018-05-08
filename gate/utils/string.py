# Copyright 2017 The KaiJIN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""STRING for seralize output"""

import tensorflow as tf
from gate.utils.logger import logger


def join_dots(*inputs):
  """['123', '456', '789'] -> '123.456.789'"""
  return '.'.join([item for item in inputs])


def type_list_to_str(dtype_list):
  """(type1, type2, type3)"""
  return '(' + ', '.join([item.__name__ for item in dtype_list]) + ')'


def class_members(obj):
  """return class members in string."""
  return ', '.join(['%s: %s' % item for item in sorted(obj.__dict__.items())])


def split_class_str(name):
  """construct output info"""
  return str(name).split('\'')[1].split('.')[-1] + '-> '


def print_members(obj):
  """print members"""
  common_type = [int, str, float]
  if '__dict__' in dir(obj):
    res = []
    for item in sorted(obj.__dict__.items()):
      if type(item[1]) in common_type and item[0].find('__') < 0:
        res.append('%s: %s' % item)
      elif type(item[1]) == list:
        if type(item[1][0]) in common_type:
          res.append('%s: %s' % item)
        else:
          [print_members(sub) for sub in item[1]]
      else:
        print_members(item[1])
    if len(res):
      logger.cfg(split_class_str(obj.__class__) + ', '.join(sorted(res)))


def as_batch(tensor, batchsize):
  """convert tensor to string type"""
  if tensor.dtype == 'string':
    return tensor
  return tf.as_string(tf.reshape(tensor, shape=[batchsize]))


def concat(batchsize, tensorlist, connector=' '):
  """t1 + connector + t2 + connector + ... + tn
  Note:
    1) convert tensor to batch string
    2) combine them with connector
  """
  str_batch = [as_batch(i, batchsize) for i in tensorlist]
  return tf.string_join(str_batch, connector)
