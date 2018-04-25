# -*- coding: utf-8 -*-
"""
DECTRION FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2017/6/21

--------------------------------------------------------

Softmax Loss OPS

"""

import tensorflow as tf
from gate.env import env


def loss(logit, label, config):
  """ Computes sparse softmax cross entropy between logits and labels.
    Measures the probability error in discrete classification tasks in which the 
    classes are mutually exclusive (each entry is in exactly one class). 
    For example, each CIFAR-10 image is labeled with one and only one label: 
    an image can be a dog or a truck, but not both.

    NOTE: For this operation, the probability of a given label is considered 
    exclusive. That is, soft classes are not allowed, and the labels vector must 
    provide a single specific index for the true class for each row of logits 
    (each minibatch entry). For soft softmax classification with a probability 
    distribution for each entry, see softmax_cross_entropy_with_logits.

    WARNING: This op expects unscaled logits, since it performs a softmax on 
    logits internally for efficiency. Do not call this op with the output of 
    softmax, as it will produce incorrect results.

    A common use case is to have logits of shape [batch_size, num_classes] 
    and labels of shape [batch_size]. But higher dimensions are supported.
  """
  with tf.name_scope('softmax_loss'):
    # compute softmax loss
    _logit = tf.reshape(logit, [config.data.batchsize,
                                config.data.num_classes])
    _loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logit, labels=label)
    _loss = tf.reduce_mean(_loss, name='loss')
    # if save to summary
    if env._SUMMARY_SCALAR:
      tf.summary.scalar('loss', _loss)
    return _loss


def error(logit, label):
  """ return error and predication result.
  Args:
      logit: float32
      label: int
  Return:
      error: error rate for each batch
  """
  with tf.name_scope('error'):
    _logit = tf.to_float(logit)
    _predictions = tf.to_int32(tf.argmax(_logit, axis=1))
    _error = 1 - tf.reduce_mean(tf.to_float(tf.equal(_predictions, label)))
    # if save to summary
    if env._SUMMARY_SCALAR:
      tf.summary.scalar('error', _error)
    return _error, _predictions
