# -*- coding: utf-8 -*-
""" Conditional GAN
  x: real image
  y: condition
  z: random vector
"""
import tensorflow as tf
from core.network.gans.ops import *


def discriminator(batchsize, x, y, y_dim, is_training=True, reuse=False):
  with tf.variable_scope('CGAN/discriminator', reuse=reuse):
    # transform to one-hot form
    y = tf.one_hot(y, depth=y_dim, on_value=1)
    y = tf.to_float(tf.reshape(y, [batchsize, 1, 1, y_dim]))
    x = conv_cond_concat(x, y)

    net = conv2d(x, 64, (4, 4), (2, 2), name='conv1')
    net = lrelu(net, name='d_conv1')

    net = conv2d(net, 128, (4, 4), (2, 2), name='conv2')
    net = lrelu(bn(net, is_training=is_training, scope='d_bn2'))

    net = tf.reshape(net, [batchsize, -1])
    net = linear(net, 1024, scope='fc3')
    net = lrelu(bn(net, is_training=is_training, scope='d_bn3'))

    logit = linear(net, 1, scope='d_fc4')
    logit = tf.nn.sigmoid(logit)

    return logit, net


def generator(batchsize, z, y, is_training=True, reuse=False):
  with tf.variable_scope("CGAN/generator", reuse=reuse):
    # merge noise and label
    y = tf.to_float(tf.reshape(y, [batchsize, 1]))
    z = tf.concat([z, y], 1)

    net = linear(z, 1024, scope='g_fc1')
    net = tf.nn.relu(bn(net, is_training=is_training, scope='g_bn1'))

    net = linear(net, 128 * 7 * 7, scope='g_fc2')
    net = tf.nn.relu(bn(net, is_training=is_training, scope='g_bn2'))

    net = tf.reshape(net, [batchsize, 7, 7, 128])
    net = deconv2d(net, 64, (4, 4), (2, 2), name='g_dc3')
    net = tf.nn.relu(bn(net, is_training=is_training, scope='g_bn3'))

    logit = deconv2d(net, 1, (4, 4), (2, 2), name='g_dc4')
    logit = tf.nn.sigmoid(logit)

    return logit, net
