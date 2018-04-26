# -*- coding: utf-8 -*-
"""
DECTRION FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2017/6/21

--------------------------------------------------------

Some fast layer descriptors

"""

import tensorflow as tf
from tensorflow.contrib import layers


# -------------------------------------------------------
# ACTIVATION AREA
# -------------------------------------------------------

def relu(x, name='relu'):
  return tf.nn.relu(x, name)


def lrelu(x, leak=0.2, name='lrelu'):
  return tf.maximum(x, leak * x, name=name)

# -------------------------------------------------------
# POOLING AREA
# -------------------------------------------------------


def max_pool2d(x, ksize, stride, padding='SAME', name='max_pool2d'):
  return layers.max_pool2d(
      inputs=x,
      kernel_size=ksize,
      stride=stride,
      padding=padding,
      scope=name)

# -------------------------------------------------------
# NORMALIZATION AREA
# -------------------------------------------------------


def bn(x, decay=0.9, is_training=True, scope='bn'):
  return layers.batch_norm(
      inputs=x,
      decay=decay,
      updates_collections=None,
      epsilon=1e-5,
      scale=True,
      is_training=is_training,
      scope=scope)

# -------------------------------------------------------
# CONVLUTION AREA
# -------------------------------------------------------


def conv2d(x, filters, ksize, stride,
           name="conv2d", padding='SAME', reuse=None):
  return tf.layers.conv2d(
      inputs=x,
      filters=filters,
      kernel_size=ksize,
      strides=stride,
      padding=padding,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
      kernel_regularizer=layers.l2_regularizer(0.0005),
      name=name,
      reuse=reuse)


def deconv2d(x, filters, ksize, stride,
             name="deconv2d", padding='SAME', reuse=None):
  return tf.layers.conv2d_transpose(
      inputs=x,
      filters=filters,
      kernel_size=ksize,
      strides=stride,
      padding=padding,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
      name=name,
      reuse=reuse)


def conv_cond_concat(x, y):
  """Concatenate conditioning vector on feature map axis."""
  x_shapes = x.get_shape()
  y_shapes = y.get_shape()
  cond = tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])
  return tf.concat([x, y * cond], 3)


# -------------------------------------------------------
# LINEAR AREA
# -------------------------------------------------------

def linear(x, output_size, scope="linear", reuse=None):
  return layers.fully_connected(
      inputs=x,
      num_outputs=output_size,
      activation_fn=None,
      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
      biases_initializer=None,
      scope=scope,
      reuse=reuse)
