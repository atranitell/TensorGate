# -*- coding: utf-8 -*-
""" Trainable Variables
    Author: Kai JIN
    Updated: 2017-11-23
"""
import tensorflow as tf
from core.utils.logger import logger


class Variables():
  """ variables
  """

  def select_vars(self, include_name, var_list=None):
    """ from var_list choose 'include_name' and ret a list
    """
    if var_list is None:
      var_list = tf.trainable_variables()
    return [var for var in var_list if include_name in var.name]

  def exclude_vars(self, exclude_name, var_list=None):
    """ exclude variables from var list
    """
    if var_list is None:
      var_list = tf.trainable_variables()
    return [var for var in var_list if exclude_name not in var.name]

  def all(self):
    """ return all variables in the model
    """
    return tf.global_variables()

  def print_trainable_list(self):
    logger.sys('TRAINABLE LIST:')
    for var in tf.trainable_variables():
      logger.net(str(var))

  def print_global_list(self):
    logger.net('ALL VARIABLES:')
    for var in tf.global_variables():
      logger.net(str(var))

  def print_grads_list(self, grads):
    logger.sys('Gradients will be trained as list:')
    for grad, var in grads:
      logger.net(str(grad))

  def print_restore_list(self, restore_vars):
    logger.sys('RESTORE LIST:')
    for var in restore_vars:
      logger.net(str(var))

  def print_vars(self, var_list):
    logger.sys('VAR LIST:')
    for var in var_list:
      logger.net(str(var))


variables = Variables()
