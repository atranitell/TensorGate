# -*- coding: utf-8 -*-
""" updated: 2017/6/14
    basic lstm model for automatic speech recognition
"""
import tensorflow as tf
from core.network.rnn.basic_rnn import basic_rnn
from core.network.rnn.ops import *


def conv2d(x, filters, ksize, stride, name="conv2d"):
  return tf.layers.conv2d(
      inputs=x,
      filters=filters,
      kernel_size=ksize,
      strides=stride,
      padding='SAME',
      use_bias=False,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
      kernel_regularizer=layers.l2_regularizer(0.0001),
      name=name)


def fc(x, filters, name):
  return layers.fully_connected(
      x, filters,
      biases_initializer=None,
      weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
      weights_regularizer=None,
      activation_fn=None,
      scope=name)


def simplenet(x, is_training, scope='simplenet', reuse=None):
  with tf.variable_scope(scope, reuse=reuse):
    x = tf.reshape(x, [-1, 28, 28, 1])
    net = tf.nn.relu(conv2d(x, 64, (7, 7), (2, 2), name='conv1'))
    net = tf.nn.relu(conv2d(net, 128, (5, 5), (2, 2), name='conv2'))
    net = tf.nn.relu(conv2d(net, 256, (4, 4), (2, 2), name='conv3'))
    net = tf.nn.relu(conv2d(net, 256, (4, 4), (2, 2), name='conv4'))
    net = conv2d(net, 512, (2, 2), (1, 1), name='conv5')
    net = layers.dropout(net, keep_prob=0.5, is_training=is_training)
    net = tf.reduce_mean(net, [1, 2])
    net = layers.flatten(net)
    # net = fc(net, 1, 'fc1')
    return net


def cnn_brnn(inputs, config, is_training, scope='cnn_brnn'):
  """ x should have shape (batchsize, H, W, C)
  """
  with tf.variable_scope(scope):
    logit, net = basic_rnn(inputs, config, is_training)
    input_feats = []
    for idx, x in enumerate(net):
      if idx == 0:
        out = simplenet(x, is_training, reuse=False)
      else:
        out = simplenet(x, is_training, reuse=True)
      input_feats.append(out)
    input_feats = tf.transpose(tf.convert_to_tensor(input_feats), [1, 2, 0])
    net = layers.flatten(input_feats)
    net = fc(net, 1, 'fc1')
    return net, None
