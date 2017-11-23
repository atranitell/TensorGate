# -*- coding: utf-8 -*-
""" Trainable Variables
    Author: Kai JIN
    Updated: 2017-11-23
"""

import tensorflow as tf
from core.utils.logger import logger


def select_vars(include_name, var_list=None):
  """ from var_list choose 'include_name' and ret a list
  """
  if var_list is None:
    var_list = tf.trainable_variables()
  return [var for var in var_list if include_name in var.name]


def all():
  """ return all variables in the model
  """
  return tf.global_variables()


def print_trainable_list():
  logger.sys('TRAINABLE LIST:')
  for var in tf.trainable_variables():
    logger.sys(str(var))


def print_global_list():
  logger.net('ALL VARIABLES:')
  for var in tf.global_variables():
    logger.net(str(var))


def print_grads_list(grads):
  logger.sys('Gradients will be trained as list:')
  for grad, var in grads:
    logger.sys(str(grad))


def print_restore_list(restore_vars):
  logger.sys('RESTORE LIST:')
  for var in restore_vars:
    logger.sys(str(var))
