# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
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


def conv2d(x, filters, ksize, stride, name="conv2d"):
  return tf.layers.conv2d(
      inputs=x,
      filters=filters,
      kernel_size=ksize,
      strides=stride,
      padding='SAME',
      kernel_initializer=layers.xavier_initializer(),
      kernel_regularizer=layers.l2_regularizer(0.0001),
      name=name)


def lrelu(x, leak=0.2):
  return tf.maximum(x, leak * x)


def simplenet(x, num_classes, is_training, scope='simplenet'):
  with tf.variable_scope(scope):
    end_points = {}
    # 112
    net = conv2d(x, 64, (7, 7), (2, 2), name='conv1')
    net = lrelu(bn(net, is_training, 'bn_conv1'))
    # 56
    net = conv2d(net, 128, (5, 5), (2, 2), name='conv2')
    net = lrelu(bn(net, is_training, 'bn_conv2'))
    # 28
    net = conv2d(net, 128, (4, 4), (2, 2), name='conv3')
    net = lrelu(bn(net, is_training, 'bn_conv3'))
    # 14
    net = conv2d(net, 256, (4, 4), (2, 2), name='conv4')
    net = lrelu(bn(net, is_training, 'bn_conv4'))
    # 7
    net = conv2d(net, 512, (4, 4), (2, 2), name='conv5')
    net = tf.reduce_mean(net, [1, 2])
    net = layers.flatten(net)

    logits = layers.fully_connected(
        net, num_classes,
        biases_initializer=tf.zeros_initializer(),
        weights_initializer=tf.truncated_normal_initializer(
            stddev=1 / 512.0),
        weights_regularizer=None,
        activation_fn=None,
        scope='logits')

    end_points['logits'] = logits
    return logits, end_points
