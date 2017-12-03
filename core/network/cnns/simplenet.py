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
from core.utils.logger import logger


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
      kernel_regularizer=None,  # layers.l2_regularizer(0.00001),
      name=name)


def sign(x):
  return tf.sign(x)


def lrelu(x, leak=0.2):
  return tf.maximum(x, leak * x)


def block(x, fileter, is_training, name):
  with tf.variable_scope(name):
    net = conv2d(x, fileter * 2, 1, 1, 'conv1')
    net = lrelu(bn(net, is_training, 'bn_conv1'))
    net = conv2d(x, fileter, 3, 1, 'conv2')
    net = lrelu(bn(net, is_training, 'bn_conv2'))
    net = conv2d(x, fileter, 1, 1, 'conv3')
    net = lrelu(bn(net, is_training, 'bn_conv3'))
    return net + x


def simplenet(x, num_classes, is_training):
  """     | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
  --------|---|---|---|---|---|---|---|
  bn      | x | x | x | o | o | o | o |
  block   | x | x | x | o | o | o | o |
  sign    | x | o | o | x | x | x | x |
  relu    | o | x | x | o | o | o | o |
  lrelu   | x | x | x | x | x | x | x |
  pool    | x | x | x | x | o | o | o |
  dropout | o | o | x | x | x | x | o |
  fc      | x | x | x | x | x | o | x |

  1) sign function seems to be worse
  """
  return simplenet_5(x, num_classes, is_training)


def simplenet_1(x, num_classes, is_training, scope='simplenet1'):
  """
  """
  with tf.variable_scope(scope):
    end_points = {}
    # 112
    net = tf.nn.relu(conv2d(x, 64, (7, 7), (2, 2), name='conv1'))
    # 56
    net = tf.nn.relu(conv2d(net, 128, (5, 5), (2, 2), name='conv2'))
    # 28
    net = tf.nn.relu(conv2d(net, 256, (4, 4), (2, 2), name='conv3'))
    # 14
    net = tf.nn.relu(conv2d(net, 256, (4, 4), (2, 2), name='conv4'))
    # 7
    net = layers.dropout(net, keep_prob=0.5, is_training=is_training)
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


def simplenet_2(x, num_classes, is_training, scope='simplenet2'):
  with tf.variable_scope(scope):
    end_points = {}
    # 112
    net = sign(conv2d(x, 64, (7, 7), (2, 2), name='conv1'))
    # 56
    net = sign(conv2d(net, 128, (5, 5), (2, 2), name='conv2'))
    # 28
    net = sign(conv2d(net, 256, (4, 4), (2, 2), name='conv3'))
    # 14
    net = sign(conv2d(net, 256, (4, 4), (2, 2), name='conv4'))
    # 7
    net = layers.dropout(net, keep_prob=0.5, is_training=is_training)
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


def simplenet_3(x, num_classes, is_training, scope='simplenet3'):
  with tf.variable_scope(scope):
    end_points = {}
    # 112
    net = sign(conv2d(x, 64, (7, 7), (2, 2), name='conv1'))
    # 56
    net = sign(conv2d(net, 128, (5, 5), (2, 2), name='conv2'))
    # 28
    net = sign(conv2d(net, 256, (4, 4), (2, 2), name='conv3'))
    # 14
    net = sign(conv2d(net, 256, (4, 4), (2, 2), name='conv4'))
    # 7
    # net = layers.dropout(net, keep_prob=0.5, is_training=is_training)
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


def simplenet_4(x, num_classes, is_training, scope='simplenet4'):
  with tf.variable_scope(scope):
    end_points = {}
    # 112
    net = conv2d(x, 64, (7, 7), (2, 2), name='conv1')
    net = tf.nn.relu(bn(net, is_training, 'bn_conv1'))
    net = block(net, 64, is_training, 'b1')
    # 56
    net = conv2d(net, 128, (5, 5), (2, 2), name='conv2')
    net = tf.nn.relu(bn(net, is_training, 'bn_conv2'))
    net = block(net, 128, is_training, 'b2')
    # 28
    net = conv2d(net, 256, (4, 4), (2, 2), name='conv3')
    net = tf.nn.relu(bn(net, is_training, 'bn_conv3'))
    net = block(net, 256, is_training, 'b3')
    # 14
    net = conv2d(net, 256, (4, 4), (2, 2), name='conv4')
    net = tf.nn.relu(bn(net, is_training, 'bn_conv4'))
    net = block(net, 256, is_training, 'b4')
    # 7
    # net = layers.dropout(net, keep_prob=0.5, is_training=is_training)
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


def simplenet_5(x, num_classes, is_training, scope='simplenet5'):
  with tf.variable_scope(scope):
    end_points = {}
    # 112
    net = conv2d(x, 64, (7, 7), (1, 1), name='conv1')
    net = layers.max_pool2d(net, [3, 3], 2, padding='SAME')
    net = tf.nn.relu(bn(net, is_training, 'bn_conv1'))
    net = block(net, 64, is_training, 'b1')
    # 56
    net = conv2d(net, 128, (5, 5), (1, 1), name='conv2')
    net = layers.max_pool2d(net, [3, 3], 2, padding='SAME')
    net = tf.nn.relu(bn(net, is_training, 'bn_conv2'))
    net = block(net, 128, is_training, 'b2')
    # 28
    net = conv2d(net, 256, (4, 4), (1, 1), name='conv3')
    net = layers.max_pool2d(net, [3, 3], 2, padding='SAME')
    net = tf.nn.relu(bn(net, is_training, 'bn_conv3'))
    net = block(net, 256, is_training, 'b3')
    # 14
    net = conv2d(net, 256, (4, 4), (1, 1), name='conv4')
    net = layers.max_pool2d(net, [3, 3], 2, padding='SAME')
    net = tf.nn.relu(bn(net, is_training, 'bn_conv4'))
    net = block(net, 256, is_training, 'b4')
    # 7
    # net = layers.dropout(net, keep_prob=0.5, is_training=is_training)
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


def simplenet_6(x, num_classes, is_training, scope='simplenet6'):
  with tf.variable_scope(scope):
    end_points = {}
    # 112
    net = conv2d(x, 64, (7, 7), (1, 1), name='conv1')
    net = layers.max_pool2d(net, [3, 3], 2, padding='SAME')
    net = tf.nn.relu(bn(net, is_training, 'bn_conv1'))
    net = block(net, 64, is_training, 'b1')
    # 56
    net = conv2d(net, 128, (5, 5), (1, 1), name='conv2')
    net = layers.max_pool2d(net, [3, 3], 2, padding='SAME')
    net = tf.nn.relu(bn(net, is_training, 'bn_conv2'))
    net = block(net, 128, is_training, 'b2')
    # 28
    net = conv2d(net, 256, (4, 4), (1, 1), name='conv3')
    net = layers.max_pool2d(net, [3, 3], 2, padding='SAME')
    net = tf.nn.relu(bn(net, is_training, 'bn_conv3'))
    net = block(net, 256, is_training, 'b3')
    # 14
    net = conv2d(net, 256, (4, 4), (1, 1), name='conv4')
    net = layers.max_pool2d(net, [3, 3], 2, padding='SAME')
    net = tf.nn.relu(bn(net, is_training, 'bn_conv4'))
    net = block(net, 256, is_training, 'b4')
    # 7
    # net = layers.dropout(net, keep_prob=0.5, is_training=is_training)
    net = conv2d(net, 512, (4, 4), (2, 2), name='conv5')
    net = tf.reduce_mean(net, [1, 2])
    net = layers.flatten(net)

    # fc1
    net = layers.fully_connected(
        net, 1024,
        biases_initializer=tf.zeros_initializer(),
        weights_initializer=tf.truncated_normal_initializer(
            stddev=1 / 512.0),
        weights_regularizer=None,
        activation_fn=tf.nn.relu,
        scope='fc1')

    logits = layers.fully_connected(
        net, num_classes,
        biases_initializer=tf.zeros_initializer(),
        weights_initializer=tf.truncated_normal_initializer(
            stddev=1 / 1024.0),
        weights_regularizer=None,
        activation_fn=None,
        scope='logits')

    end_points['logits'] = logits
    return logits, end_points
