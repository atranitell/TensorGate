# Copyright 2017 The KaiJIN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""FOR KINFACE VAE-GAN"""

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import nn


def lrelu(x, leak=0.2, name='lrelu'):
  return tf.maximum(x, leak * x, name=name)


def conv(x, num_filters, k_size, k_stride, name=None, weight_decay=0.0001):
  return tf.layers.conv1d(
      inputs=x,
      filters=num_filters,
      kernel_size=k_size,
      strides=k_stride,
      padding='SAME',
      activation=None,
      dilation_rate=1,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
      kernel_regularizer=layers.l2_regularizer(weight_decay),
      bias_initializer=tf.constant_initializer(0.0),
      name=name)


def deconv(x, num_filters, k_size, k_stride, name=None):
  n, w, c = x.get_shape().as_list()
  filter = tf.Variable(tf.constant(1.0, dtype=tf.float32,
                                   shape=[k_size, num_filters, c]))
  return nn.conv1d_transpose(
      value=x,
      filter=filter,
      output_shape=tf.convert_to_tensor([n, w*k_stride, num_filters]),
      stride=k_stride,
      padding='SAME',
      data_format='NWC',
      name=name)


def pool(inputs, kernel_size, strides,
         pooling_type='MAX', padding_type='SAME', name=None):
  return tf.nn.pool(
      inputs,
      window_shape=[kernel_size],
      pooling_type=pooling_type,
      padding=padding_type,
      strides=[strides],
      name=name)


def encoder(inputs, num_classes, is_training, reuse=None):
  with tf.variable_scope("SenFCN/encoder", reuse=reuse):
    # inputs-6400
    net = lrelu(conv(inputs, 64, 20, 2, name='conv1'))
    net = pool(net, 2, 2, name='pool1')
    # output-1600
    net = lrelu(conv(net, 128, 10, 2, 'conv2'))
    net = pool(net, 2, 2, name='pool2')
    # output-400
    net = lrelu(conv(net, 128, 10, 2, 'conv3'))
    net = pool(net, 2, 2, name='pool3')
    # output-100
    net = lrelu(conv(net, 256, 10, 2, 'conv4'))
    # output-50
    net = tf.reduce_sum(net, axis=2)
    # output-256
    # net = layers.dropout(net, 0.5, is_training=is_training)
    logits = layers.fully_connected(
        net, num_classes,
        biases_initializer=None,
        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
        weights_regularizer=None,
        activation_fn=None,
        scope='logits')
    return logits, net


def decoder(inputs, is_training, reuse=None):
  with tf.variable_scope("SenFCN/decoder", reuse=reuse):
    net = tf.expand_dims(inputs, 2)
    net = lrelu(deconv(net, 256, 10, 2, name='deconv1'))
    # inputs=100
    net = lrelu(deconv(net, 128, 10, 4, name='deconv2'))
    net = lrelu(deconv(net, 128, 10, 4, name='deconv3'))
    net = lrelu(deconv(net, 64, 10, 4, name='deconv4'))
    net = lrelu(deconv(net, 1, 10, 1, name='deconv5'))
    return net
    
# def encoder(x, z_dim, is_training=True, reuse=None):
#   with tf.variable_scope("SenVAE/encoder", reuse=reuse):    # condition
#     # y = tf.one_hot(y, depth=y_dim, on_value=1)
#     # y = tf.to_float(tf.reshape(y, [-1, 1, 1, y_dim]))
#     # x = conv_cond_concat(x, y)
#     # network
#     net = conv(x, 128, 3, 1, is_training, 'conv1', False)
#     net = conv(net, 128, 5, 2, is_training, 'conv2')
#     net = conv(net, 256, 5, 2, is_training, 'conv3')
#     net = conv(net, 512, 5, 2, is_training, 'conv4')
#     net = layers.flatten(tf.reduce_mean(net, [1, 2]))
#     # output
#     gaussian_params = linear(net, 2 * z_dim, scope='fc1')
#     mean = gaussian_params[:, :z_dim]
#     stddev = 1e-8 + tf.nn.softplus(gaussian_params[:, z_dim:])
#     return mean, stddev, net


# def generator(z, y, is_training=True, reuse=None):
#   with tf.variable_scope("KIN_VAE/generator", reuse=reuse):
#     # merge noise and label
#     # y = tf.to_float(tf.reshape(y, [-1, 1]))
#     # z = tf.concat([z, y], 1)
#     # reshape to 2d
#     net = linear(z, 512 * 1 * 1, scope='fc1')
#     net = lrelu(bn(net, is_training=is_training, scope='bn_fc1'))
#     net = tf.reshape(net, [-1, 1, 1, 512])
#     # network
#     net = deconv(net, 512, 5, 1, is_training, 'deconv1')
#     net = deconv(net, 256, 5, 2, is_training, 'deconv2')
#     net = deconv(net, 128, 5, 2, is_training, 'deconv3')
#     net = deconv(net, 128, 5, 2, is_training, 'deconv4')
#     net = deconv(net, 3, 4, 1, is_training, 'logit', False, False)
#     # output
#     logit = tf.nn.sigmoid(net)
#     return logit


# def discriminator(x, y, y_dim, is_training, reuse=None):
#   with tf.variable_scope('KIN_VAE/discriminator', reuse=reuse):
#     # condition
#     # y = tf.one_hot(y, depth=y_dim, on_value=1)
#     # y = tf.to_float(tf.reshape(y, [-1, 1, 1, y_dim]))
#     # x = conv_cond_concat(x, y)
#     # network
#     net = conv(x, 128, 3, 1, is_training, 'conv1', False)
#     net = conv(net, 128, 5, 2, is_training, 'conv2')
#     net = conv(net, 256, 5, 2, is_training, 'conv3')
#     net = conv(net, 512, 5, 2, is_training, 'conv4')
#     net = layers.flatten(tf.reduce_mean(net, [1, 2]))
#     return net
