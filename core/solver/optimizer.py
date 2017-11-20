# -*- coding: utf-8 -*-
""" 
    Author: Kai JIN
    Updated: 2017-03-16
"""
import tensorflow as tf
from core.utils.logger import logger


def configure(config, learning_rate):
  """Configures the optimizer used for training.
  Args:
    config: config['train']['optimizer']
    learning_rate: A scalar or `Tensor` learning rate.
  Returns:
    An instance of an optimizer.
  Raises:
    ValueError: if opt.optimizer is not recognized.
  """

  logger.info('Routine will use %s optimizer.' % config['type'])
  cfg = config[config['type']]

  if config['type'] == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=cfg['rho'],
        epsilon=cfg['epsilon'])

  elif config['type'] == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=cfg['accumulator_value'])

  elif config['type'] == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=cfg['beta1'],
        beta2=cfg['beta2'],
        epsilon=cfg['epsilon'])

  elif config['type'] == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=cfg['power'],
        initial_accumulator_value=cfg['accumulator_value'],
        l1_regularization_strength=cfg['l1'],
        l2_regularization_strength=cfg['l2'])

  elif config['type'] == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=cfg['momentum'],
        name='Momentum')

  elif config['type'] == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=cfg['decay'],
        momentum=cfg['momentum'],
        epsilon=cfg['epsilon'])

  elif config['type'] == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

  else:
    raise ValueError('Optimizer [%s] was not recognized' % config['type'])

  return optimizer
