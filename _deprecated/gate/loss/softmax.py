# -*- coding: utf-8 -*-
""" loss function for Softmax on classification
"""
import tensorflow as tf


def get_loss(logits, labels, num_classes, batch_size):
    """ softmax with loss
    Return:
        _losses: a scalr with float32
        _logits: a copy of logits with float32
        labels: return a raw input
    """
    with tf.name_scope('softmax_loss'):
        _logits = tf.reshape(logits, [batch_size, num_classes])
        _losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=_logits, name='loss_batch')
        _losses = tf.reduce_mean(_losses, name='loss')
        tf.summary.scalar('losses', _losses)
        return _losses, labels, _logits


def get_error(logits, labels):
    """ return mae and rmse value
    Args:
        logits: float32
        labels: float32 and identical scale with logits
    Return:
        err: error rate for each batch
    """
    with tf.name_scope('error'):
        predictions = tf.to_int32(tf.argmax(logits, axis=1))
        err = 1 - tf.reduce_mean(tf.to_float(tf.equal(predictions, labels)))
        tf.summary.scalar('err', err)
        return err, predictions
