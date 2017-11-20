# -*- coding: utf-8 -*-
""" Learning rate assemble.
    Author: Kai JIN
    Updated: 2017-07-23
"""
import tensorflow as tf


def decay_step(num_epochs_per_decay, batchsize, total_num):
  return int(total_num / batchsize * num_epochs_per_decay)


def configure(config, global_step, batchsize, total_num):
  """Configures the learning rate.
  Args:
    config: config['train']['lr']
    global_step: The global_step tensor.
  Returns:
    A `Tensor` representing the learning rate.
  """
  method = config['type']

  if config['type'] == 'exponential':
    return tf.train.exponential_decay(
        config['learning_rate'],
        global_step,
        decay_step(config[method]['num_epochs_per_decay'],
                   batchsize, total_num),
        config[method]['decay_factor'],
        staircase=True,
        name='exponential_decay_learning_rate')

  elif config['type'] == 'fixed':
    return tf.constant(config['learning_rate'], name='fixed_learning_rate')

  elif config['type'] == 'vstep':
    global_step = tf.to_int32(global_step)
    return tf.train.piecewise_constant(
        global_step,
        config[method]['boundaries'],
        config[method]['values'],
        name='vstep_decay_learning_rate')

  elif config['type'] == 'polynomial':
    return tf.train.polynomial_decay(
        config['learning_rate'],
        global_step,
        decay_step(config[method]['num_epochs_per_decay'],
                   batchsize, total_num),
        config[method]['end_learning_rate'],
        power=1.0,
        cycle=False,
        name='polynomial_decay_learning_rate')

  elif config['type'] == 'natural_exp':
    return tf.train.natural_exp_decay(
        config['learning_rate'],
        global_step,
        decay_step(config[method]['num_epochs_per_decay'],
                   batchsize, total_num),
        config[method]['decay_rate'],
        staircase=True,
        name='natural_exp_decay_learning_rate')
  else:
    raise ValueError('learning rate type [ % s] was not recognized' % (
                     config['type']))
