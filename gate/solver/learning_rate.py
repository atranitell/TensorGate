# -*- coding: utf-8 -*-
"""
GATE FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2017/7/23

--------------------------------------------------------

Learning Rate

"""

import tensorflow as tf


def decay_steps(total_num, batchsize, decay_epochs):
  return int(total_num / batchsize * decay_epochs)


def configure_lr(config, global_step, batchsize, total_num):
  """Configures the learning rate.
  Args:
    config: config.train.lr
    global_step: The global_step tensor.
  Returns:
    A `Tensor` representing the learning rate.
  """
  if config.name == 'exponential':
    return tf.train.exponential_decay(
        config.learning_rate,
        global_step,
        decay_steps(total_num, batchsize, config.decay_epochs),
        config.decay_factor,
        staircase=False,
        name='exponential_decay_learning_rate')

  elif config.name == 'fixed':
    return tf.constant(config.learning_rate, name='fixed_learning_rate')

  elif config.name == 'vstep':
    global_step = tf.to_int32(global_step)
    return tf.train.piecewise_constant(
        global_step,
        config.boundaries,
        config.values,
        name='vstep_decay_learning_rate')

  elif config.name == 'polynomial':
    return tf.train.polynomial_decay(
        config.learning_rate,
        global_step,
        decay_steps(total_num, batchsize, config.decay_epochs),
        config.end_learning_rate,
        power=1.0,
        cycle=False,
        name='polynomial_decay_learning_rate')

  elif config.name == 'natural_exp':
    return tf.train.natural_exp_decay(
        config.learning_rate,
        global_step,
        decay_steps(total_num, batchsize, config.decay_epochs),
        config.decay_rate,
        staircase=True,
        name='natural_exp_decay_learning_rate')

  elif config.name == 'cosine':
    return tf.train.cosine_decay(
        config.learning_rate,
        global_step,
        decay_steps(total_num, batchsize, config.decay_epochs),
        name="cosine_decay_learning_rate")

  elif config.name == 'linear_cosine':
    return tf.train.linear_cosine_decay(
        config.learning_rate,
        global_step,
        decay_steps(total_num, batchsize, config.decay_epochs),
        name="linear_cosine_decay_learning_rate")

  elif config.name == 'noisy_linear_cosine':
    return tf.train.noisy_linear_cosine_decay(
        config.learning_rate,
        global_step,
        decay_steps(total_num, batchsize, config.decay_epochs),
        name="noisy_linear_cosine_decay_learning_rate")

  elif config.name == 'inverse_time':
    return tf.train.inverse_time_decay(
        config.learning_rate,
        global_step,
        decay_steps(total_num, batchsize, config.decay_epochs),
        config.decay_rate,
        name="inverse_time_decay_learning_rate")

  else:
    error = 'learning rate type [ % s] was not recognized' % (config.name)
    raise ValueError(error)
