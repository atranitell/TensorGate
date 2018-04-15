# -*- coding: utf-8 -*-
"""
Most codes from https://github.com/carpedm20/DCGAN-tensorflow
"""
import math
import tensorflow as tf
from tensorflow.contrib import layers


def bn(x, is_training, scope):
  return layers.batch_norm(
      inputs=x,
      decay=0.9,
      updates_collections=None,
      epsilon=1e-5,
      scale=True,
      is_training=is_training,
      scope=scope)


def conv_cond_concat(x, y):
  """Concatenate conditioning vector on feature map axis."""
  x_shapes = x.get_shape()
  y_shapes = y.get_shape()
  cond = tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])
  return tf.concat([x, y * cond], 3)


def conv2d(x, filters, ksize, stride, name="conv2d", padding='SAME'):
  return tf.layers.conv2d(
      inputs=x,
      filters=filters,
      kernel_size=ksize,
      strides=stride,
      padding=padding,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
      kernel_regularizer=layers.l2_regularizer(0.0005),
      name=name)


def deconv2d(x, filters, ksize, stride, name="deconv2d", padding='SAME'):
  return tf.layers.conv2d_transpose(
      inputs=x,
      filters=filters,
      kernel_size=ksize,
      strides=stride,
      padding=padding,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
      name=name)


def lrelu(x, leak=0.2, name=None):
  return tf.maximum(x, leak * x)


def linear(x, output_size, scope="linear"):
  return layers.fully_connected(
      inputs=x,
      num_outputs=output_size,
      activation_fn=None,
      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
      scope=scope)