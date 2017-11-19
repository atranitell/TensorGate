# -*- coding: utf-8 -*-
""" updated: 2017/3/16
"""
import tensorflow as tf
from gate.utils.logger import logger


def configure(opt, learning_rate):
    """Configures the optimizer used for training.
    Args:
      learning_rate: A scalar or `Tensor` learning rate.
    Returns:
      An instance of an optimizer.
    Raises:
      ValueError: if opt.optimizer is not recognized.
    """

    logger.info('Routine will use %s optimizer.' % opt.optimizer)

    if opt.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=opt.adadelta_rho,
            epsilon=opt.adadelta_epsilon)

    elif opt.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=opt.adagrad_initial_accumulator_value)

    elif opt.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=opt.adam_beta1,
            beta2=opt.adam_beta2,
            epsilon=opt.adam_epsilon)

    elif opt.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=opt.ftrl_learning_rate_power,
            initial_accumulator_value=opt.ftrl_initial_accumulator_value,
            l1_regularization_strength=opt.ftrl_l1,
            l2_regularization_strength=opt.ftrl_l2)

    elif opt.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=opt.momentum,
            name='Momentum')

    elif opt.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=opt.rmsprop_decay,
            momentum=opt.rmsprop_momentum,
            epsilon=opt.rmsprop_epsilon)

    elif opt.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    else:
        raise ValueError('Optimizer [%s] was not recognized' % opt.optimizer)

    return optimizer
