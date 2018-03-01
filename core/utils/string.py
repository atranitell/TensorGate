# -*- coding: utf-8 -*-
""" String operation
    Author: Kai JIN
    Updated: 2017-06-23
"""
import tensorflow as tf
from core.utils.logger import logger


class String():
  """ string operation
  """

  def clip_last_sub_string(self, content, separator='/', keep_sep=False):
    """ raw: a/b/c/d/e
        return: if keep_sep is true -> a/b/c/d/ else -> a/b/c/d
    """
    st = str(content).split(separator)
    nw = ''
    for i in range(len(st) - 1):
      if keep_sep is True and i == len(st) - 2:
        nw += st[i]
      else:
        nw += st[i] + separator
    return nw

  def type_list_to_str(self, dtype_list):
    """ (type1, type2, type3)
    """
    return '(' + ', '.join([item.__name__ for item in dtype_list]) + ')'

  def class_members(self, obj):
    """ return class members in string.
    """
    return ', '.join(['%s: %s' % item for item in sorted(obj.__dict__.items())])

  def split_class_str(self, name):
    """ construct output info
    """
    return str(name).split('\'')[1].split('.')[2] + '-> '

  def print_members(self, obj):
    """ print members
    """
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
            [self.print_members(sub) for sub in item[1]]
        else:
          self.print_members(item[1])
      if len(res):
        logger.info(self.split_class_str(
            obj.__class__) + ', '.join(sorted(res)))

  def as_batch(self, tensor, batchsize):
    """ convert tensor to string type
    """
    if tensor.dtype == 'string':
      return tensor
    return tf.as_string(tf.reshape(tensor, shape=[batchsize]))

  def concat(self, batchsize, tensorlist, connector=' '):
    """ t1 + connector + t2 + connector + ... + tn
      1) convert tensor to batch string
      2) combine them with connector
    """
    str_batch = [self.as_batch(i, batchsize) for i in tensorlist]
    return tf.string_join(str_batch, connector)


string = String()
