# -*- coding: utf-8 -*-
""" 
    Author: Kai JIN
    Updated: 2017-03-16
"""
import tensorflow as tf
from core.utils.logger import logger


def configure_optimizer(config, learning_rate):
  """Configures the optimizer used for training.
  Args:
    config: config.train.optimizer
    learning_rate: A scalar or `Tensor` learning rate.
  Returns:
    An instance of an optimizer.
  Raises:
    ValueError: if opt.optimizer is not recognized.
  """
  logger.info('Routine will use %s optimizer.' % config.name)

  if config.name == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=config.rho,
        epsilon=config.epsilon)

  elif config.name == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=config.initial_accumulator_value)

  elif config.name == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=config.beta1,
        beta2=config.beta2,
        epsilon=config.epsilon)

  elif config.name == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=config.learning_rate_power,
        initial_accumulator_value=config.initial_accumulator_value,
        l1_regularization_strength=config.l1,
        l2_regularization_strength=config.l2)

  elif config.name == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=config.momentum,
        name='Momentum')

  elif config.name == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=config.decay,
        momentum=config.momentum,
        epsilon=config.epsilon)

  elif config.name == 'proximal':
    optimizer = tf.train.ProximalGradientDescentOptimizer(
        learning_rate,
        l1_regularization_strength=config.l1_regularization_strength,
        l2_regularization_strength=config.l2_regularization_strength)

  elif config.name == 'proximal_adagrad':
    optimizer = tf.train.ProximalAdagradOptimizer(
        learning_rate,
        initial_accumulator_value=config.initial_accumulator_value,
        l1_regularization_strength=config.l1_regularization_strength,
        l2_regularization_strength=config.l2_regularization_strength)

  elif config.name == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

  else:
    raise ValueError('Optimizer [%s] was not recognized' % config.name)

  return optimizer
