# -*- coding: utf-8 -*-
""" loss function for cosine distance
"""
import tensorflow as tf


def get_loss(x, y, labels, batch_size, is_training=True):
  """ cosine loss: 1-<x , y>/(norm(x)*norm(y))*label
      label is +1 / -1
  Args:
      scale: true for scale labels from [0, num_classes] to [0, 1]
  Return:
      losses: a scalr with float32
      loss: a batchsize of loss for per sample
  """
  with tf.name_scope('cosine_loss'):
    labels = tf.reshape(labels, [batch_size, 1])
    norm_x = tf.reshape(tf.norm(x, axis=1), [batch_size, 1])
    norm_y = tf.reshape(tf.norm(y, axis=1), [batch_size, 1])

    x1 = tf.expand_dims(x, axis=2)
    x1 = tf.transpose(x1, perm=[0, 2, 1])
    y1 = tf.expand_dims(y, axis=2)
    
    loss = tf.reshape(
        tf.matmul(x1, y1), [batch_size, 1]) / (norm_x * norm_y)

    if is_training:
      loss = loss * tf.to_float(labels)

    losses = 1.0 - tf.reduce_mean(loss)
    loss = 1 - loss

    tf.summary.scalar('losses', losses)
    return losses, loss
