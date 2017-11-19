# -*- coding: utf-8 -*-
""" learning rate assemble.
"""
import tensorflow as tf


def configure(lr, global_step):
    """Configures the learning rate.
    Args:
      lr: dataset.lr parameters
      global_step: The global_step tensor.
    Returns:
      A `Tensor` representing the learning rate.
    """

    if lr.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(lr.learning_rate,
                                          global_step,
                                          lr.decay_step,
                                          lr.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')

    elif lr.learning_rate_decay_type == 'fixed':
        return tf.constant(lr.learning_rate, name='fixed_learning_rate')

    elif lr.learning_rate_decay_type == 'vstep':
        global_step = tf.to_int32(global_step)
        return tf.train.piecewise_constant(global_step,
                                           lr.boundaries,
                                           lr.values,
                                           name='vstep_decay_learning_rate')

    elif lr.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(lr.learning_rate,
                                         global_step,
                                         lr.decay_step,
                                         lr.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')

    elif lr.learning_rate_decay_type == 'natural_exp':
        return tf.train.natural_exp_decay(lr.learning_rate,
                                          global_step,
                                          lr.decay_steps,
                                          lr.decay_rate,
                                          staircase=True,
                                          name='natural_exp_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized' % (
                         lr.learning_rate_decay_type))
