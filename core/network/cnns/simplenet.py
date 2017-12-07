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
      padding='VALID',
      use_bias=False,
      kernel_initializer=layers.xavier_initializer(),
      kernel_regularizer=layers.l2_regularizer(0.0005),
      name=name)


def conv3d(x, ksize, stride, name="conv3d"):
  return tf.nn.conv3d(
      input=x,
      filter=ksize,
      strides=stride,
      padding='VALID',
      name=name)
  return tf.layers.conv3d(
      inputs=x,
      filters=filters,
      kernel_size=ksize,
      strides=stride,
      padding='VALID',
      kernel_initializer=layers.xavier_initializer(),
      kernel_regularizer=layers.l2_regularizer(0.0005),
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


def blocks(x, fileter, name, repeat=1):
  for i in range(repeat):
    with tf.variable_scope(name + '_' + str(i)):
      net = lrelu(conv2d(x, fileter * 2, 1, 1, 'conv1'))
      net = lrelu(conv2d(net, fileter, 3, 1, 'conv2'))
      net = lrelu(conv2d(net, fileter, 1, 1, 'conv3'))
    x = net + x
  return net + x


def fc(x, filters, name):
  return layers.fully_connected(
      x, filters,
      biases_initializer=None,
      weights_initializer=layers.xavier_initializer(),
      weights_regularizer=None,
      activation_fn=None,
      scope=name)


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
  return simplenet_12(x, num_classes, is_training)


def simplenet_11(x, num_classes, is_training, scope='simplenet11'):
  """
  """
  with tf.variable_scope(scope):
    end_points = {}
    net = lrelu(conv2d(x, 64, (7, 7), (2, 2), name='conv1'))
    net = lrelu(conv2d(net, 128, (5, 5), (2, 2), name='conv2'))
    net = lrelu(conv2d(net, 256, (4, 4), (2, 2), name='conv3'))
    net = lrelu(conv2d(net, 256, (4, 4), (2, 2), name='conv4'))
    net = conv2d(net, 512, (4, 4), (2, 2), name='conv5')
    net = layers.dropout(net, keep_prob=0.5, is_training=is_training)
    net = tf.reduce_mean(net, [1, 2])
    net = layers.flatten(net)
    logits = fc(net, num_classes, 'logits')
    return logits, end_points


def simplenet_13(x, num_classes, is_training, scope='simplenet13'):
  """
  """
  with tf.variable_scope(scope):
    end_points = {}
    net = lrelu(conv2d(x, 64, (7, 7), (1, 1), name='conv1'))
    net = lrelu(conv2d(x, 128, (7, 7), (2, 2), name='conv1'))
    net = lrelu(conv2d(net, 256, (5, 5), (1, 1), name='conv2'))
    net = lrelu(conv2d(net, 256, (5, 5), (2, 2), name='conv2'))
    net = lrelu(conv2d(net, 512, (4, 4), (1, 1), name='conv3'))
    net = lrelu(conv2d(net, 512, (4, 4), (2, 2), name='conv4'))
    net = conv2d(net, 512, (4, 4), (2, 2), name='conv5')
    net = layers.dropout(net, keep_prob=0.5, is_training=is_training)
    net = tf.reduce_mean(net, [1, 2])
    net = layers.flatten(net)
    logits = fc(net, num_classes, 'logits')
    return logits, end_points


def simplenet_12(x, num_classes, is_training, scope='simplenet12'):
  """
  """
  with tf.variable_scope(scope):
    end_points = {}
    net = lrelu(conv2d(x, 64, (7, 7), (1, 1), name='conv1'))
    net = tf.layers.max_pooling2d(net, 2, 2, name='pool1')
    net = lrelu(conv2d(net, 128, (5, 5), (1, 1), name='conv2'))
    net = tf.layers.max_pooling2d(net, 2, 2, name='pool2')
    net = lrelu(conv2d(net, 256, (4, 4), (1, 1), name='conv3'))
    net = tf.layers.max_pooling2d(net, 2, 2, name='pool3')
    net = lrelu(conv2d(net, 256, (4, 4), (1, 1), name='conv4'))
    net = tf.layers.max_pooling2d(net, 2, 2, name='pool4')
    net = conv2d(net, 512, (3, 3), (1, 1), name='conv5')
    net = layers.dropout(net, keep_prob=0.5, is_training=is_training)
    net = tf.reduce_mean(net, [1, 2])
    net = layers.flatten(net)
    logits = fc(net, num_classes, 'logits')
    return logits, end_points


# def simplenet_2(x, num_classes, is_training, scope='simplenet2'):
#   with tf.variable_scope(scope):
#     end_points = {}
#     # 112
#     net = sign(conv2d(x, 64, (7, 7), (2, 2), name='conv1'))
#     # 56
#     net = sign(conv2d(net, 128, (5, 5), (2, 2), name='conv2'))
#     # 28
#     net = sign(conv2d(net, 256, (4, 4), (2, 2), name='conv3'))
#     # 14
#     net = sign(conv2d(net, 256, (4, 4), (2, 2), name='conv4'))
#     # 7
#     net = layers.dropout(net, keep_prob=0.5, is_training=is_training)
#     net = conv2d(net, 512, (4, 4), (2, 2), name='conv5')
#     net = tf.reduce_mean(net, [1, 2])
#     net = layers.flatten(net)

#     logits = layers.fully_connected(
#         net, num_classes,
#         biases_initializer=tf.zeros_initializer(),
#         weights_initializer=tf.truncated_normal_initializer(
#             stddev=1 / 512.0),
#         weights_regularizer=None,
#         activation_fn=None,
#         scope='logits')

#     end_points['logits'] = logits
#     return logits, end_points


# def simplenet_3(x, num_classes, is_training, scope='simplenet3'):
#   with tf.variable_scope(scope):
#     end_points = {}
#     # 112
#     net = sign(conv2d(x, 64, (7, 7), (2, 2), name='conv1'))
#     # 56
#     net = sign(conv2d(net, 128, (5, 5), (2, 2), name='conv2'))
#     # 28
#     net = sign(conv2d(net, 256, (4, 4), (2, 2), name='conv3'))
#     # 14
#     net = sign(conv2d(net, 256, (4, 4), (2, 2), name='conv4'))
#     # 7
#     # net = layers.dropout(net, keep_prob=0.5, is_training=is_training)
#     net = conv2d(net, 512, (4, 4), (2, 2), name='conv5')
#     net = tf.reduce_mean(net, [1, 2])
#     net = layers.flatten(net)

#     logits = layers.fully_connected(
#         net, num_classes,
#         biases_initializer=tf.zeros_initializer(),
#         weights_initializer=tf.truncated_normal_initializer(
#             stddev=1 / 512.0),
#         weights_regularizer=None,
#         activation_fn=None,
#         scope='logits')

#     end_points['logits'] = logits
#     return logits, end_points


# def simplenet_4(x, num_classes, is_training, scope='simplenet4'):
#   with tf.variable_scope(scope):
#     end_points = {}
#     # 112
#     net = lrelu(conv2d(x, 64, (7, 7), (2, 2), name='conv1'))
#     net = blocks(net, 64, 'b1', 1)
#     # 56
#     net = lrelu(conv2d(net, 128, (5, 5), (2, 2), name='conv2'))
#     net = blocks(net, 128, 'b2', 3)
#     # 28
#     net = lrelu(conv2d(net, 256, (4, 4), (2, 2), name='conv3'))
#     net = blocks(net, 256, 'b3', 3)
#     # 14
#     net = lrelu(conv2d(net, 256, (4, 4), (2, 2), name='conv4'))
#     net = blocks(net, 256, 'b4', 1)
#     # 7
#     net = conv2d(net, 512, (4, 4), (2, 2), name='conv5')
#     # f
#     net = layers.dropout(net, keep_prob=0.5, is_training=is_training)
#     net = tf.reduce_mean(net, [1, 2])
#     net = layers.flatten(net)
#     logits = layers.fully_connected(
#         net, num_classes,
#         biases_initializer=tf.zeros_initializer(),
#         weights_initializer=tf.truncated_normal_initializer(
#             stddev=1 / 512.0),
#         weights_regularizer=None,
#         activation_fn=None,
#         scope='logits')
#     end_points['logits'] = logits
#     return logits, end_points


# def simplenet_5(x, num_classes, is_training, scope='simplenet5'):
#   with tf.variable_scope(scope):
#     end_points = {}
#     # 112
#     net = lrelu(conv2d(x, 64, (7, 7), (2, 2), name='conv1'))
#     net = blocks(net, 64, 'b1', 1)
#     # 56
#     net = lrelu(conv2d(net, 128, (5, 5), (2, 2), name='conv2'))
#     net = blocks(net, 128, 'b2', 1)
#     # 28
#     net = lrelu(conv2d(net, 256, (4, 4), (2, 2), name='conv3'))
#     net = blocks(net, 256, 'b3', 1)
#     # 14
#     net = lrelu(conv2d(net, 256, (4, 4), (2, 2), name='conv4'))
#     net = blocks(net, 256, 'b4', 1)
#     # 7
#     net = conv2d(net, 512, (4, 4), (2, 2), name='conv5')
#     # f
#     net = layers.dropout(net, keep_prob=0.5, is_training=is_training)
#     net = tf.reduce_mean(net, [1, 2])
#     net = layers.flatten(net)
#     logits = layers.fully_connected(
#         net, num_classes,
#         biases_initializer=tf.zeros_initializer(),
#         weights_initializer=tf.truncated_normal_initializer(
#             stddev=1 / 512.0),
#         weights_regularizer=None,
#         activation_fn=None,
#         scope='logits')
#     end_points['logits'] = logits
#     return logits, end_points


# def simplenet_6(x, num_classes, is_training, scope='simplenet6'):
#   """
#   """
#   with tf.variable_scope(scope):
#     end_points = {}
#     # 112
#     net = tf.nn.relu(conv2d(x, 64, (56, 56), (2, 2), name='conv1'))
#     # 56
#     net = tf.nn.relu(conv2d(net, 128, (28, 28), (2, 2), name='conv2'))
#     # 28
#     net = tf.nn.relu(conv2d(net, 256, (14, 14), (2, 2), name='conv3'))
#     # 14
#     net = tf.nn.relu(conv2d(net, 256, (7, 7), (2, 2), name='conv4'))
#     # 7
#     net = conv2d(net, 512, (5, 5), (2, 2), name='conv5')
#     net = layers.dropout(net, keep_prob=0.5, is_training=is_training)
#     net = tf.reduce_mean(net, [1, 2])
#     net = layers.flatten(net)

#     logits = layers.fully_connected(
#         net, num_classes,
#         biases_initializer=tf.zeros_initializer(),
#         weights_initializer=tf.truncated_normal_initializer(
#             stddev=1 / 512.0),
#         weights_regularizer=None,
#         activation_fn=None,
#         scope='logits')

#     end_points['logits'] = logits
#     return logits, end_points


# def simplenet_7(x, num_classes, is_training, scope='simplenet7'):
#   """
#   """
#   with tf.variable_scope(scope):
#     end_points = {}
#     x = tf.reshape(x, (-1, 10, 112, 112, 1))
#     # x = tf.transpose(x, [0, 3, 1, 2, 4])
#     print(x)

#     # filter: A `Tensor`. Must have the same type as `input`.
#     #   Shape `[filter_depth, filter_height, filter_width, in_channels,
#     #   out_channels]`. `in_channels` must match between `input` and `filter`.
#     # strides: A list of `ints` that has length `>= 5`.
#     #   1-D tensor of length 5. The stride of the sliding window for each
#     #   dimension of `input`. Must have `strides[0] = strides[4] = 1`.

#     # 112
#     net = tf.nn.relu(conv3d(x, [2, 7, 7, 1, 64], [1, 2, 2, 1, 1], name='conv1'))
#     print(net)
#     # 56
#     net = tf.nn.relu(conv3d(net, [2, 5, 5, 64, 128], [1, 2, 2, 1, 1], name='conv2'))
#     print(net)
#     # 28
#     net = tf.nn.relu(conv3d(net, [2, 4, 4, 128, 256], [1, 2, 2, 1, 1], name='conv3'))
#     print(net)
#     # 14
#     net = tf.nn.relu(conv3d(net, [2, 4, 4, 256, 256], [1, 2, 2, 1, 1], name='conv4'))
#     print(net)
#     # 7
#     net = layers.dropout(net, keep_prob=0.5, is_training=is_training)
#     net = conv3d(net, [2, 4, 4, 256, 512], [1, 2, 2, 1, 1], name='conv5')
#     print(net)
#     net = tf.reduce_mean(net, [1, 2, 3])
#     net = layers.flatten(net)

#     logits = layers.fully_connected(
#         net, num_classes,
#         biases_initializer=tf.zeros_initializer(),
#         weights_initializer=tf.truncated_normal_initializer(
#             stddev=1 / 512.0),
#         weights_regularizer=None,
#         activation_fn=None,
#         scope='logits')

#     end_points['logits'] = logits
#     return logits, end_points
