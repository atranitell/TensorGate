# -*- coding: utf-8 -*-
""" CVAE-GAN: Fine-Grained Image Generation through Asymmetric Training
    https://arxiv.org/abs/1703.10155
    updated: 2017/11/24
"""
import tensorflow as tf
from tensorflow.contrib import layers


def deconv2d(x, filters, ksize=(5, 5), stride=(2, 2), stddev=0.02, name="deconv2d"):
  return tf.layers.conv2d_transpose(
      inputs=x,
      filters=filters,
      kernel_size=ksize,
      strides=stride,
      padding='SAME',
      kernel_initializer=layers.xavier_initializer(),
      name=name)


def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak * x)


def linear(x, output_size, scope="linear"):
  return layers.fully_connected(
      inputs=x,
      num_outputs=output_size,
      activation_fn=None,
      weights_initializer=layers.xavier_initializer(),
      scope=scope)


def bn(x, is_training, scope):
  return layers.batch_norm(
      inputs=x,
      decay=0.9,
      updates_collections=None,
      epsilon=1e-5,
      scale=True,
      is_training=is_training,
      scope=scope)


def conv2d(x, filters, ksize=(5, 5), stride=(2, 2), stddev=0.02, name="conv2d"):
  return tf.layers.conv2d(
      inputs=x,
      filters=filters,
      kernel_size=ksize,
      strides=stride,
      padding='SAME',
      kernel_initializer=layers.xavier_initializer(),
      kernel_regularizer=layers.l2_regularizer(0.0005),
      name=name)


def encoder(x, z_dim, is_training=True, reuse=None):
  with tf.variable_scope("KIN_VAE/encoder", reuse=reuse):
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
    # net = linear(net, 256, scope='fc7')
    # net = lrelu(bn(net, is_training=is_training, scope='en_bn7'))
    # output gaussian distribution
    gaussian_params = linear(net, 2 * z_dim, scope='en_fc8')
    # The mean parameter is unconstrained
    mean = gaussian_params[:, :z_dim]
    # The standard deviation must be positive. Parametrize with a softplus and
    # add a small epsilon for numerical stability
    stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, z_dim:])
    return mean, stddev, net


def generator(z, is_training=True, reuse=None):
  with tf.variable_scope("KIN_VAE/generator", reuse=reuse):
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


def discriminator(x, is_training, reuse=None):
  with tf.variable_scope('KIN_VAE/discriminator', reuse=reuse):
    # input 1x64x64
    net = conv2d(x, 64, (4, 4), (2, 2), name='conv1')
    net = lrelu(net, name='d_conv1')
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


def recognitor(x, is_training, scope, reuse=None):
  with tf.variable_scope('KIN_VAE/recognitor/' + scope, reuse=reuse):
    # input 1x64x64
    net = conv2d(x, 64, (4, 4), (2, 2), name='conv1')
    net = lrelu(bn(net, is_training, 'r_bn1'))
    # input 64x32x32
    net = conv2d(net, 128, (4, 4), (2, 2), name='conv2')
    net = lrelu(bn(net, is_training, 'r_bn2'))
    # input 128x16x16
    net = conv2d(net, 128, (4, 4), (2, 2), name='conv3')
    net = lrelu(bn(net, is_training, 'r_bn3'))
    # input 128x8x8
    net = conv2d(net, 256, (4, 4), (2, 2), name='conv4')
    net = lrelu(bn(net, is_training, 'r_bn4'))
    # input 256x4x4
    net = conv2d(net, 256, (4, 4), (2, 2), name='conv5')
    net = lrelu(bn(net, is_training, 'r_bn5'))
    # input 256x2x2
    net = conv2d(net, 512, (4, 4), (2, 2), name='conv6')
    # input 512x1x1
    # for cosine - output features
    net = layers.flatten(net)
    return net
