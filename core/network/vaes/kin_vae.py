# -*- coding: utf-8 -*-
""" FOR KINFACE VAE-GAN
    updated: 2017/11/24
"""
import tensorflow as tf
from tensorflow.contrib import layers
from core.network.gans.ops import *


def encoder(x, y, y_dim, z_dim, is_training=True, reuse=None):
  with tf.variable_scope("KIN_VAE/encoder", reuse=reuse):
    # condition
    y = tf.one_hot(y, depth=y_dim, on_value=1)
    y = tf.to_float(tf.reshape(y, [-1, 1, 1, y_dim]))
    x = conv_cond_concat(x, y)
    # input 3x64x64
    net = conv2d(x, 64, (4, 4), (2, 2), name='conv1')
    net = lrelu(net, name='en_conv1')
    # input 64x32x32
    net = conv2d(net, 128, (4, 4), (2, 2), name='conv2')
    net = lrelu(bn(net, is_training=is_training, scope='en_bn2'))
    # input 128x16x16
    net = conv2d(net, 128, (4, 4), (2, 2), name='conv3')
    net = lrelu(bn(net, is_training=is_training, scope='en_bn3'))
    # input 256x8x8
    net = conv2d(net, 256, (4, 4), (2, 2), name='conv4')
    net = lrelu(bn(net, is_training=is_training, scope='en_bn4'))
    # input 256x4x4
    net = conv2d(net, 256, (4, 4), (2, 2), name='conv5')
    net = lrelu(bn(net, is_training=is_training, scope='en_bn5'))
    # input 256x2x2
    net = conv2d(net, 512, (4, 4), (2, 2), name='conv6')
    # net = lrelu(bn(net, is_training=is_training, scope='en_bn6'))
    # input 512x1x1
    net = layers.flatten(net)
    # net = lrelu(bn(net, is_training=is_training, scope='en_bn7'))
    # output gaussian distribution
    gaussian_params = linear(net, 2 * z_dim, scope='en_fc8')
    # The mean parameter is unconstrained
    mean = gaussian_params[:, :z_dim]
    # The standard deviation must be positive. Parametrize with a softplus and
    # add a small epsilon for numerical stability
    stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, z_dim:])
    return mean, stddev, net


def generator(z, y, is_training=True, reuse=None):
  with tf.variable_scope("KIN_VAE/generator", reuse=reuse):
    # merge noise and label
    y = tf.to_float(tf.reshape(y, [-1, 1]))
    z = tf.concat([z, y], 1)
    # input z_dim
    net = linear(z, 512 * 1 * 1, scope='de_fc1')
    net = lrelu(bn(net, is_training=is_training, scope='de_bn1'))
    net = tf.reshape(net, [-1, 1, 1, 512])
    # input 512*1*1
    net = deconv2d(net, 256, (4, 4), (2, 2), name='de_dc2')
    net = lrelu(bn(net, is_training=is_training, scope='de_bn2'))
    # input 256*2*2
    net = deconv2d(net, 256, (4, 4), (2, 2), name='de_dc3')
    net = lrelu(bn(net, is_training=is_training, scope='de_bn3'))
    # input 256*4*4
    net = deconv2d(net, 128, (4, 4), (2, 2), name='de_dc4')
    net = lrelu(bn(net, is_training=is_training, scope='de_bn4'))
    # input 128*8*8
    net = deconv2d(net, 128, (4, 4), (2, 2), name='de_dc5')
    net = lrelu(bn(net, is_training=is_training, scope='de_bn5'))
    # input 128*16*16
    net = deconv2d(net, 64, (4, 4), (2, 2), name='de_dc6')
    net = lrelu(bn(net, is_training=is_training, scope='de_bn6'))
    # input 64*32*32
    logit = deconv2d(net, 3, (4, 4), (2, 2), name='de_dc7')
    # input 1*64*64
    logit = tf.nn.sigmoid(logit)
    return logit


def discriminator(x, y, y_dim, is_training, reuse=None):
  with tf.variable_scope('KIN_VAE/discriminator', reuse=reuse):
    # condition
    y = tf.one_hot(y, depth=y_dim, on_value=1)
    y = tf.to_float(tf.reshape(y, [-1, 1, 1, y_dim]))
    x = conv_cond_concat(x, y)
    # input 1x64x64
    net = conv2d(x, 64, (4, 4), (2, 2), name='conv1')
    # input 64x32x32
    net = conv2d(net, 128, (4, 4), (2, 2), name='conv2')
    net = lrelu(bn(net, is_training=is_training, scope='d_bn2'))
    # input 128x16x16
    net = conv2d(net, 128, (4, 4), (2, 2), name='conv3')
    net = lrelu(bn(net, is_training=is_training, scope='d_bn3'))
    # input 256x8x8
    net = conv2d(net, 256, (4, 4), (2, 2), name='conv4')
    net = lrelu(bn(net, is_training=is_training, scope='d_bn4'))
    # input 256x4x4
    net = conv2d(net, 256, (4, 4), (2, 2), name='conv5')
    net = lrelu(bn(net, is_training=is_training, scope='d_bn5'))
    # input 256x2x2
    net = conv2d(net, 512, (4, 4), (2, 2), name='conv6')
    # input 512x1x1
    net = layers.flatten(net)
    # input 1024
    # logit = tf.nn.sigmoid(net)
    return net, net
