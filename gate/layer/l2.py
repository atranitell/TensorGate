# -*- coding: utf-8 -*-
"""
DECTRION FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2017/6/21

--------------------------------------------------------

L2 LOSS OPS

"""

import tensorflow as tf
from gate.env import env


def loss(logits, labels, config):
  # range, batchsize, scale=True
  """ l2 loss: |labels - logits|^2
  Args:
      scale: true for scale labels from [0, range] to [0, 1]
  Return:
      _losses: a scalr with float32
      _logits: a copy of logits with float32
      _labels: to float32 and scaled to [0, 1]
  """
  with tf.name_scope('l2_loss'):
    _logits = tf.to_float(tf.reshape(logits, [config.data.batchsize, 1]))
    _labels = tf.to_float(tf.reshape(labels, [config.data.batchsize, 1]))
    if config.data.scale:
      _labels = tf.divide(_labels, config.data.range)
    _losses = tf.nn.l2_loss([_labels - _logits], name='loss')
    if env._SUMMARY_SCALAR:
      tf.summary.scalar('losses', _losses)
    return _losses


def error(logits, labels, config):
  """ return mae and rmse value
  Args:
      logits: float32
      labels: float32 and identical scale with logits
  Return:
      mae: mean absolute error
      rmse: root of mean square error
  """
  with tf.name_scope('error'):
    _logits = tf.to_float(tf.reshape(logits, [config.data.batchsize, 1]))
    _labels = tf.to_float(tf.reshape(labels, [config.data.batchsize, 1]))
    ranges = 1 if not config.data.scale else config.data.range
    err_mae = tf.reduce_mean(input_tensor=tf.abs(
        (_logits - _labels) * ranges), name='error_mae')
    err_rmse = tf.sqrt(tf.reduce_mean(input_tensor=tf.square(
        (_logits - _labels) * ranges), name='error_mse'))
    if env._SUMMARY_SCALAR:
      tf.summary.scalar('mae', err_mae)
      tf.summary.scalar('rmse', err_rmse)
    return err_mae, err_rmse
