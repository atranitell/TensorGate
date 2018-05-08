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
"""Trainable Variable"""

import tensorflow as tf
from gate.utils.logger import logger


def select_vars(include_name, var_list=None):
  """from var_list choose 'include_name' and ret a list"""
  if var_list is None:
    var_list = tf.trainable_variables()
  return [var for var in var_list if include_name in var.name]


def exclude_vars(exclude_name, var_list=None):
  """exclude variables from var list"""
  if var_list is None:
    var_list = tf.trainable_variables()
  return [var for var in var_list if exclude_name not in var.name]


def all():
  """return all variables in the model"""
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
