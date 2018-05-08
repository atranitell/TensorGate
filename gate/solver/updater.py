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
"""UPDATER"""

import tensorflow as tf
from gate.utils import variable
from gate.utils.logger import logger
from gate.solver.optimizer import configure_optimizer
from gate.solver.learning_rate import configure_lr
from gate.env import env


def default(config, loss, global_step, var_list=None, index=0):
  """ For an updater, it should include:
  1) configure learning rate
  2) configure optimizer
  3) gradients ops like clip

  Considering multi-updater, each should independently train,
    it should receive var list as input.

  Args:
    var_list: variables should be trained.
    index: a flag to pinpoint setting. For single updater, the value is 0.

  """
  # default to train all variables in the network.
  if var_list is None:
    var_list = tf.trainable_variables()
  else:
    logger.info('Specify training vars list:')
    variable.print_vars(var_list)

  # configure learning rate
  lr = configure_lr(config=config.lr[index],
                    global_step=tf.train.get_global_step(),
                    batchsize=config.data.batchsize,
                    total_num=config.data.total_num)
  tf.summary.scalar('train/lr', lr)

  # configure optimizer
  optimizer = configure_optimizer(config.opt[index], lr)

  # add batch norm to update collections
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    grads = optimizer.compute_gradients(loss, var_list=var_list)
    grad_op = optimizer.apply_gradients(grads, global_step)

  # assemble
  train_op = grad_op

  # add
  add_grad_to_summary(grads, env._SUMMARY_GRAD_STAT, env._SUMMARY_GRAD_HIST)
  add_weight_to_summary(env._SUMMARY_WEIGHT_STAT, env._SUMMARY_WEIGHT_HIST)

  # info
  logger.sys('Updater has been initialized.')

  return train_op


def add_grad_to_summary(grads, grad_summary=True, grad_hist=False):
  """ save grad info
  """
  if grad_summary is False and grad_hist is False:
    logger.sys('GRAD is NOT recorded.')
    return
  with tf.name_scope('grads'):
    for grad, var in grads:
      prefix = var.op.name
      if prefix.find('global_step') == 0 or grad is None:
        continue
      if grad_summary:
        tf.summary.scalar(var.op.name + '_mean', tf.reduce_mean(grad))
        tf.summary.scalar(var.op.name + '_max', tf.reduce_max(grad))
        tf.summary.scalar(var.op.name + '_sum', tf.reduce_sum(grad))
      if grad_hist:
        tf.summary.histogram(var.op.name + '/gradients', grad)


def add_weight_to_summary(weight_summary=True, weight_hist=False):
  """ save weight info
  """
  if weight_summary is False and weight_hist is False:
    logger.sys('WEIGHT is NOT recorded.')
    return
  with tf.name_scope('weights'):
    for weight in tf.trainable_variables():
      prefix = weight.op.name
      if prefix.find('global_step') == 0 or weight is None:
        continue
      if weight_summary:
        tf.summary.scalar(weight.op.name + '_mean', tf.reduce_mean(weight))
        tf.summary.scalar(weight.op.name + '_max', tf.reduce_max(weight))
        tf.summary.scalar(weight.op.name + '_sum', tf.reduce_sum(weight))
      # Add histograms for trainable variables.
      if weight_hist:
        tf.summary.histogram(weight.op.name, weight)
