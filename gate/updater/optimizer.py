# -*- coding: utf-8 -*-
""" updated: 2017/3/16
"""

import tensorflow as tf


def configure(dataset, learning_rate):
    """Configures the optimizer used for training.

    Args:
      learning_rate: A scalar or `Tensor` learning rate.

    Returns:
      An instance of an optimizer.

    Raises:
      ValueError: if dataset.opt.optimizer is not recognized.
    """
    print('[TRAIN] Routine will use %s optimizer.' % dataset.opt.optimizer)

    if dataset.opt.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=dataset.opt.adadelta_rho,
            epsilon=dataset.opt.opt_epsilon)

    elif dataset.opt.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=dataset.opt.adagrad_initial_accumulator_value)

    elif dataset.opt.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=dataset.opt.adam_beta1,
            beta2=dataset.opt.adam_beta2,
            epsilon=dataset.opt.opt_epsilon)

    elif dataset.opt.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=dataset.opt.ftrl_learning_rate_power,
            initial_accumulator_value=dataset.opt.ftrl_initial_accumulator_value,
            l1_regularization_strength=dataset.opt.ftrl_l1,
            l2_regularization_strength=dataset.opt.ftrl_l2)

    elif dataset.opt.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=dataset.opt.momentum,
            name='Momentum')

    elif dataset.opt.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=dataset.opt.rmsprop_decay,
            momentum=dataset.opt.rmsprop_momentum,
            epsilon=dataset.opt.opt_epsilon)

    elif dataset.opt.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    else:
        raise ValueError(
            'Optimizer [%s] was not recognized', dataset.opt.optimizer)

    return optimizer
