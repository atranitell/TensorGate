# -*- coding: utf-8 -*-
""" FOR KINFACE VAE-GAN
    updated: 2017/11/24
"""
import tensorflow as tf
from tensorflow.contrib import layers
from core.network.vae.ops import *


def conv(inputs, num, kernel, stride, is_training, name,
         with_bn=True, with_activation=True):
  net = conv2d(inputs, num, (kernel, kernel),
               (stride, stride), name=name, padding='VALID')
  if with_bn:
    net = bn(net, is_training=is_training, scope='bn_' + name)
  if with_activation:
    net = lrelu(net)
  return net


def deconv(inputs, num, kernel, stride, is_training, name,
           with_bn=True, with_activation=True):
  net = deconv2d(inputs, num, (kernel, kernel),
                 (stride, stride), name=name, padding='VALID')
  if with_bn:
    net = bn(net, is_training=is_training, scope='bn_' + name)
  if with_activation:
    net = lrelu(net)
  return net


def encoder(x, z_dim, is_training=True, reuse=None):
  with tf.variable_scope("KIN_VAE/encoder", reuse=reuse):    # condition
    # y = tf.one_hot(y, depth=y_dim, on_value=1)
    # y = tf.to_float(tf.reshape(y, [-1, 1, 1, y_dim]))
    # x = conv_cond_concat(x, y)
    # network
    net = conv(x, 128, 3, 1, is_training, 'conv1', False)
    net = conv(net, 128, 5, 2, is_training, 'conv2')
    net = conv(net, 256, 5, 2, is_training, 'conv3')
    net = conv(net, 512, 5, 2, is_training, 'conv4')
    net = layers.flatten(tf.reduce_mean(net, [1, 2]))
    # output
    gaussian_params = linear(net, 2 * z_dim, scope='fc1')
    mean = gaussian_params[:, :z_dim]
    stddev = 1e-8 + tf.nn.softplus(gaussian_params[:, z_dim:])
    return mean, stddev, net


def generator(z, y, is_training=True, reuse=None):
  with tf.variable_scope("KIN_VAE/generator", reuse=reuse):
    # merge noise and label
    y = tf.to_float(tf.reshape(y, [-1, 1]))
    z = tf.concat([z, y], 1)
    # reshape to 2d
    net = linear(z, 512 * 1 * 1, scope='fc1')
    net = lrelu(bn(net, is_training=is_training, scope='bn_fc1'))
    net = tf.reshape(net, [-1, 1, 1, 512])
    # network
    net = deconv(net, 512, 5, 1, is_training, 'deconv1')
    net = deconv(net, 256, 5, 2, is_training, 'deconv2')
    net = deconv(net, 128, 5, 2, is_training, 'deconv3')
    net = deconv(net, 128, 5, 2, is_training, 'deconv4')
    net = deconv(net, 3, 4, 1, is_training, 'logit', False, False)
    # output
    logit = tf.nn.sigmoid(net)
    return logit


def discriminator(x, y, y_dim, is_training, reuse=None):
  with tf.variable_scope('KIN_VAE/discriminator', reuse=reuse):
    # condition
    y = tf.one_hot(y, depth=y_dim, on_value=1)
    y = tf.to_float(tf.reshape(y, [-1, 1, 1, y_dim]))
    x = conv_cond_concat(x, y)
    # network
    net = conv(x, 128, 3, 1, is_training, 'conv1', False)
    net = conv(net, 128, 5, 2, is_training, 'conv2')
    net = conv(net, 256, 5, 2, is_training, 'conv3')
    net = conv(net, 512, 5, 2, is_training, 'conv4')
    net = layers.flatten(tf.reduce_mean(net, [1, 2]))
    return net
