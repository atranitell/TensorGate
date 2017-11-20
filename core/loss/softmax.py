# -*- coding: utf-8 -*-
""" Loss function for Softmax on classification.
    Author: Kai JIN
    Updated: 2017-06-21
"""
import tensorflow as tf


def get_loss(logits, labels, num_classes, batch_size):
    """ softmax with loss
    Return:
        loss: a scalr with float32
        logit: a copy of logits with float32
    """
    with tf.name_scope('softmax_loss'):
        logit = tf.reshape(logits, [batch_size, num_classes])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logit, name='loss_batch')
        loss = tf.reduce_mean(loss, name='loss')
        tf.summary.scalar('losses', loss)
        return loss, logit


def get_error(logits, labels):
    """ return mae and rmse value
    Args:
        logits: float32
        labels: float32 and identical scale with logits
    Return:
        error: error rate for each batch
    """
    with tf.name_scope('error'):
        predictions = tf.to_int32(tf.argmax(logits, axis=1))
        error = 1 - tf.reduce_mean(tf.to_float(tf.equal(predictions, labels)))
        tf.summary.scalar('error', error)
        return error, predictions
