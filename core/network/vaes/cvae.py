# -*- coding: utf-8 -*-
""" Conditional GAN
  x: real image
  y: condition
  z: random vector
"""
import tensorflow as tf
from core.network.gans.ops import *


# Gaussian Encoder
def encoder(batchsize, x, y, y_dim, z_dim,
            is_training=True, reuse=False):
  with tf.variable_scope("CAVE/encoder", reuse=reuse):
    # transform to one-hot form
    y = tf.one_hot(y, depth=y_dim, on_value=1)
    y = tf.to_float(tf.reshape(y, [batchsize, 1, 1, y_dim]))
    x = conv_cond_concat(x, y)

    net = conv2d(x, 64, (4, 4), (2, 2), name='conv1')
    net = lrelu(net, name='en_conv1')

    net = conv2d(net, 128, (4, 4), (2, 2), name='conv2')
    net = lrelu(bn(net, is_training=is_training, scope='en_bn2'))

    net = tf.reshape(net, [batchsize, -1])
    net = linear(net, 1024, scope='fc3')
    net = lrelu(bn(net, is_training=is_training, scope='en_bn3'))

    gaussian_params = linear(net, 2 * z_dim, scope='en_fc4')

    # The mean parameter is unconstrained
    mean = gaussian_params[:, :z_dim]
    # The standard deviation must be positive. Parametrize with a softplus and
    # add a small epsilon for numerical stability
    stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, z_dim:])

    return mean, stddev


def decoder(batchsize, z, y, is_training=True, reuse=False):
  # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
  # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
  # Bernoulli decoder
  with tf.variable_scope("CAVE/decoder", reuse=reuse):

    # merge noise and label
    y = tf.to_float(tf.reshape(y, [batchsize, 1]))
    z = tf.concat([z, y], 1)

    net = linear(z, 1024, scope='de_fc1')
    net = tf.nn.relu(bn(net, is_training=is_training, scope='de_bn1'))

    net = linear(net, 128 * 7 * 7, scope='de_fc2')
    net = tf.nn.relu(bn(net, is_training=is_training, scope='de_bn2'))

    net = tf.reshape(net, [batchsize, 7, 7, 128])
    net = deconv2d(net, 64, (4, 4), (2, 2), name='de_dc3')
    net = tf.nn.relu(bn(net, is_training=is_training, scope='de_bn3'))

    logit = deconv2d(net, 1, (4, 4), (2, 2), name='de_dc4')
    logit = tf.nn.sigmoid(logit)

    return logit