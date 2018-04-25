# -*- coding: utf-8 -*-
"""
DECTRION FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2018/4/21

--------------------------------------------------------

Trainable Variable

"""

import tensorflow as tf
from gate.util.logger import logger


def select_vars(include_name, var_list=None):
  """ from var_list choose 'include_name' and ret a list
  """
  if var_list is None:
    var_list = tf.trainable_variables()
  return [var for var in var_list if include_name in var.name]


def exclude_vars(exclude_name, var_list=None):
  """ exclude variables from var list
  """
  if var_list is None:
    var_list = tf.trainable_variables()
  return [var for var in var_list if exclude_name not in var.name]


def all():
  """ return all variables in the model
  """
  return tf.global_variables()


def print_trainable_list():
  logger.sys('TRAINABLE LIST:')
  for var in tf.trainable_variables():
    logger.net(str(var))


def print_global_list():
  logger.net('ALL VARIABLES:')
  for var in tf.global_variables():
    logger.net(str(var))


def print_grads_list(grads):
  logger.sys('Gradients will be trained as list:')
  for grad, var in grads:
    logger.net(str(grad))


def print_restore_list(restore_vars):
  logger.sys('RESTORE LIST:')
  for var in restore_vars:
    logger.net(str(var))


def print_vars(var_list):
  logger.sys('VAR LIST:')
  for var in var_list:
    logger.net(str(var))