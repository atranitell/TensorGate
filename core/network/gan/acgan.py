# -*- coding: utf-8 -*-
""" Conditional Image Synthesis With Auxiliary Classifier GANs
    https://arxiv.org/abs/1610.09585
    updated: 2017/11/22
"""
import tensorflow as tf
from core.network.gan.ops import *


def classifier(x, y_dim, is_training, reuse):
  with tf.variable_scope("ACGAN/classifier", reuse=reuse):
    net = linear(x, 128, scope='c_fc1')
    net = lrelu(bn(net, is_training=is_training, scope='c_bn1'))
    logit = linear(net, y_dim, scope='c_fc2')
    out = tf.nn.softmax(logit)

    return out


def discriminator(batchsize, x, is_training=True, reuse=False):
  with tf.variable_scope('ACGAN/discriminator', reuse=reuse):

    net = conv2d(x, 64, (4, 4), (2, 2), name='conv1')
    net = lrelu(net, name='d_conv1')

    net = conv2d(net, 128, (4, 4), (2, 2), name='conv2')
    net = lrelu(bn(net, is_training=is_training, scope='d_bn2'))

    net = tf.reshape(net, [batchsize, -1])
    net = linear(net, 1024, scope='fc3')
    net = lrelu(bn(net, is_training=is_training, scope='d_bn3'))

    logit = linear(net, 1, scope='d_fc4')
    out = tf.nn.sigmoid(logit)

    return logit, net, out


def generator(batchsize, z, y, is_training=True, reuse=False):
  with tf.variable_scope("ACGAN/generator", reuse=reuse):
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

    logit = deconv2d(net, 3, (4, 4), (2, 2), name='g_dc4')
    logit = tf.nn.sigmoid(logit)

    return logit, net