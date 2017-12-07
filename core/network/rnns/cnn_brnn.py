# -*- coding: utf-8 -*-
""" updated: 2017/6/14
    basic lstm model for automatic speech recognition
"""
import tensorflow as tf
from core.network.rnns.basic_rnn import basic_rnn
from core.network.rnns.ops import *


def conv2d(x, filters, ksize, stride, name="conv2d"):
  return tf.layers.conv2d(
      inputs=x,
      filters=filters,
      kernel_size=ksize,
      strides=stride,
      padding='VALID',
      use_bias=False,
      kernel_initializer=layers.xavier_initializer(),
      kernel_regularizer=layers.l2_regularizer(0.0001),
      name=name)


def simplenet(x, is_training, scope='simplenet', reuse=None):
  with tf.variable_scope(scope, reuse=reuse):
    x = tf.reshape(x, x.get_shape().as_list() + [1])
    net = tf.nn.relu(conv2d(x, 64, (7, 7), (2, 2), name='conv1'))
    net = tf.nn.relu(conv2d(net, 128, (5, 5), (2, 2), name='conv2'))
    net = tf.nn.relu(conv2d(net, 256, (4, 4), (2, 2), name='conv3'))
    net = tf.nn.relu(conv2d(net, 256, (4, 4), (2, 2), name='conv4'))
    net = conv2d(net, 512, (4, 4), (2, 2), name='conv5')
    net = layers.dropout(net, keep_prob=0.5, is_training=is_training)
    net = tf.reduce_mean(net, [1, 2])
    net = layers.flatten(net)
    return net


def cnn_brnn(inputs, config, is_training, scope='cnn_brnn'):
  """ x should have shape (batchsize, H, W, C)
  """
  with tf.variable_scope(scope):
    input_list = tf.unstack(inputs, axis=3)
    input_feats = []
    for idx, x in enumerate(input_list):
      if idx == 0:
        net = simplenet(x, is_training, reuse=False)
      else:
        net = simplenet(x, is_training, reuse=True)
      input_feats.append(net)
    input_feats = tf.transpose(tf.convert_to_tensor(input_feats), [1, 2, 0])
    logit, net = basic_rnn(input_feats, config, is_training)
    return logit, net
