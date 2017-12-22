# -*- coding: utf-8 -*-
""" Updater
    Author: Kai JIN
    Updated: 2017-08-28
"""
import tensorflow as tf
from core.utils.logger import logger
from core.solver.optimizer import configure_optimizer
from core.solver.learning_rate import configure_lr


def default(config, loss, global_step, var_list=None, index=0):
  """ For an updater, it should include:
  1) configure learning rate
  2) configure optimizer
  3) gradients ops like clip

  Considering multi-updater, each should independently train,
    it should receive var list as input.

  var_list: variables should be trained.
  index: a flag to pinpoint setting. For single updater, the value is 0.
  """
  # default to train all variables in the network.
  if var_list == None:
    var_list = tf.trainable_variables()

  # configure learning rate
  lr = configure_lr(config.lr[index],
                    tf.train.get_global_step(),
                    config.data.batchsize,
                    config.data.total_num)
  tf.summary.scalar('train/lr', lr)

  # configure optimizer
  optimizer = configure_optimizer(config.optimizer[index], lr)

  # compute gradients
  grads = optimizer.compute_gradients(loss, var_list=var_list)

  # apply to op
  grad_op = optimizer.apply_gradients(grads, global_step)

  # assemble
  train_op = grad_op

  # add
  add_grad_to_summary(grads, True, True)
  add_weight_to_summary(True, True)

  return train_op


def add_grad_to_summary(grads, grad_summary=True, grad_hist=False):
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
