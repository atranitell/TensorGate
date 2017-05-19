# -*- coding: utf-8 -*-
""" loss function for L2 distance
"""
import tensorflow as tf


def get_loss(logits, labels, num_classes, batch_size, scale=True):
    """ l2 loss: |labels - logits|^2
    Args:
        scale: true for scale labels from [0, num_classes] to [0, 1]
    Return:
        _losses: a scalr with float32
        _logits: a copy of logits with float32
        _labels: a copy of labels with float32
    """
    with tf.name_scope('l2_loss'):
        _logits = tf.to_float(tf.reshape(logits, [batch_size, 1]))
        _labels = tf.to_float(tf.reshape(labels, [batch_size, 1]))
        if scale:
            _labels = tf.divide(_labels, num_classes)
        _losses = tf.nn.l2_loss([_labels - _logits], name='loss')

        tf.summary.scalar('losses', _losses)
        return _losses, _labels, _logits


def get_error(logits, labels, num_classes):
    """ return mae and rmse value
    Args:
        logits: float32
        labels: float32 and identical scale with logits
    Return:
        mae: mean absolute error
        rmse: root of mean square error
    """
    with tf.name_scope('error'):
        err_mae = tf.reduce_mean(input_tensor=tf.abs(
            (logits - labels) * num_classes), name='err_mae')
        err_rmse = tf.sqrt(tf.reduce_mean(input_tensor=tf.square(
            (logits - labels) * num_classes), name='err_mse'))

        tf.summary.scalar('mae', err_mae)
        tf.summary.scalar('rmse', err_rmse)
        return err_mae, err_rmse
