# -*- coding: utf-8 -*-
""" Learning rate assemble.
    Author: Kai JIN
    Updated: 2017-07-23
"""
import tensorflow as tf


def configure_lr(config, global_step):
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
        config.decay_steps,
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
        config.decay_steps,
        config.end_learning_rate,
        power=1.0,
        cycle=False,
        name='polynomial_decay_learning_rate')

  elif config.name == 'natural_exp':
    return tf.train.natural_exp_decay(
        config.learning_rate,
        global_step,
        config.decay_steps,
        config.decay_rate,
        staircase=True,
        name='natural_exp_decay_learning_rate')

  elif config.name == 'cosine':
    return tf.train.cosine_decay(
        config.learning_rate,
        global_step,
        config.decay_steps,
        name="cosine_decay_learning_rate")

  elif config.name == 'linear_cosine':
    return tf.train.linear_cosine_decay(
        config.learning_rate,
        global_step,
        config.decay_steps,
        name="linear_cosine_decay_learning_rate")

  elif config.name == 'noisy_linear_cosine':
    return tf.train.noisy_linear_cosine_decay(
        config.learning_rate,
        global_step,
        config.decay_steps,
        name="noisy_linear_cosine_decay_learning_rate")

  elif config.name == 'inverse_time':
    return tf.train.inverse_time_decay(
        config.learning_rate,
        global_step,
        config.decay_steps,
        config.decay_rate,
        name="inverse_time_decay_learning_rate")

  else:
    error = 'learning rate type [ % s] was not recognized' % (config.name)
    raise ValueError(error)
