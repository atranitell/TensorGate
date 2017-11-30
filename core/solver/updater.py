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

  # configure optimizer
  optimizer = configure_optimizer(config.optimizer[index], lr)

  # compute gradients
  grads = optimizer.compute_gradients(loss, var_list=var_list)

  # apply to op
  grad_op = optimizer.apply_gradients(grads, global_step)

  # assemble
  train_op = grad_op

  return train_op
