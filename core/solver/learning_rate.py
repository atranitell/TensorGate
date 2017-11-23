# -*- coding: utf-8 -*-
""" Learning rate assemble.
    Author: Kai JIN
    Updated: 2017-07-23
"""
import tensorflow as tf


def configure_lr(config, global_step, batchsize, total_num):
  """Configures the learning rate.
  Args:
    config: config['train']['lr']
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
  else:
    raise ValueError('learning rate type [ % s] was not recognized' % (
                     config.name))
