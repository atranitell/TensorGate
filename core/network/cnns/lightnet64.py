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
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers


def lightnet64_argscope(weight_decay,
                        batch_norm_decay,
                        batch_norm_epsilon,
                        batch_norm_scale):
  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': None,
      'zero_debias_moving_mean': True}

  with arg_scope([layers.conv2d],
                 weights_regularizer=layers.l2_regularizer(weight_decay),
                 weights_initializer=layers.xavier_initializer(),
                 #    biases_initializer=tf.constant_initializer(0.1),
                 activation_fn=tf.nn.relu,
                 normalizer_fn=None,  # layers.batch_norm,
                 #  normalizer_params=batch_norm_params,
                 padding='SAME'):
    with arg_scope([layers.batch_norm], **batch_norm_params):
      with arg_scope([layers.max_pool2d, layers.avg_pool2d], padding='SAME') as arg_sc:
        return arg_sc


def lightnet64(images, num_classes, is_training,
               dropout_keep_prob, scope='Lightnet64'):

  end_points = {}
  with tf.variable_scope(scope, 'Lightnet64', [images, num_classes]):
    with arg_scope([layers.batch_norm], is_training=is_training):
      block_in = layers.conv2d(images, 64, [7, 7], 2)

      with tf.variable_scope('block1'):
        with tf.variable_scope('branch_1_0'):
          net = layers.conv2d(block_in, 64, [1, 1], 1)
          net = layers.conv2d(net, 64, [3, 3], 1)
          branch_1_0 = layers.conv2d(net, 256, [1, 1], 1)
        net = tf.concat(axis=3, values=[branch_1_0, block_in])
        block_in = layers.conv2d(net, 128, [1, 1], 1)

      block_in = layers.max_pool2d(block_in, [3, 3], 2)
      with tf.variable_scope('block2'):
        with tf.variable_scope('branch_2_0'):
          net = layers.conv2d(block_in, 64, [1, 1], 1)
          net = layers.conv2d(net, 64, [3, 3], 1)
          branch_2_0 = layers.conv2d(net, 256, [1, 1], 1)
        net = tf.concat(axis=3, values=[branch_2_0, block_in])
        block_in = layers.conv2d(net, 256, [1, 1], 1)

      block_in = layers.max_pool2d(block_in, [3, 3], 2)

      # with tf.variable_scope('block3'):
      #   with tf.variable_scope('branch_3_0'):
      #     net = layers.conv2d(block_in, 64, [1, 1], 1)
      #     net = layers.conv2d(net, 64, [3, 3], 1)
      #     branch_3_0 = layers.conv2d(net, 256, [1, 1], 1)
      #   net = tf.concat(axis=3, values=[branch_3_0, block_in])
      #   block_in = layers.conv2d(net, 512, [1, 1], 1)

      # block_in = layers.max_pool2d(block_in, [3, 3], 2)

      # with tf.variable_scope('block4'):
      #   with tf.variable_scope('branch_4_0'):
      #     net = layers.conv2d(block_in, 64, [1, 1], 1)
      #     net = layers.conv2d(net, 64, [3, 3], 1)
      #     branch_4_0 = layers.conv2d(net, 256, [1, 1], 1)
      #   net = tf.concat(axis=3, values=[branch_4_0, block_in])
      #   block_in = layers.conv2d(net, 1024, [1, 1], 1)

    block_in = layers.dropout(
        block_in, keep_prob=dropout_keep_prob, is_training=is_training)
    block_in = layers.avg_pool2d(block_in, [7, 7], 1, padding='VALID')
    block_in = layers.flatten(block_in)

    end_points['global_pool'] = block_in

    logits = layers.fully_connected(
        block_in, num_classes,
        biases_initializer=tf.zeros_initializer(),
        weights_initializer=tf.truncated_normal_initializer(
            stddev=1 / 256.0),
        weights_regularizer=None,
        activation_fn=None,
        scope='logits')

    end_points['logits'] = logits

    return logits, end_points
