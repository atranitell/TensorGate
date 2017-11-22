# -*- coding: utf-8 -*-
""" DCGAN
  x: real image
  z: random vector
"""
import tensorflow as tf
from core.network.gans.ops import *


def discriminator(batchsize, x, is_training=True, reuse=False):
  with tf.variable_scope('DCGAN/discriminator', reuse=reuse):
    # transform to one-hot form
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


def generator(batchsize, z, is_training=True, reuse=False):
  with tf.variable_scope("DCGAN/generator", reuse=reuse):
    # output
    net = linear(z, 1024, scope='g_fc1')
    net = tf.nn.relu(bn(net, is_training=is_training, scope='g_bn1'))

    net = linear(net, 128 * 7 * 7, scope='g_fc2')
    net = tf.nn.relu(bn(net, is_training=is_training, scope='g_bn2'))

    net = tf.reshape(net, [batchsize, 7, 7, 128])
    net = deconv2d(net, 64, (4, 4), (2, 2), name='g_dc3')
    net = tf.nn.relu(bn(net, is_training=is_training, scope='g_bn3'))

    logit = deconv2d(net, 3, (4, 4), (2, 2), name='g_dc4')
    logit = tf.nn.sigmoid(logit)

    return logit, net
