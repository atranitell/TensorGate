# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""ResNet model.

Related papers:
https://arxiv.org/pdf/1603.05027v2.pdf
https://arxiv.org/pdf/1512.03385v1.pdf
https://arxiv.org/pdf/1605.07146v1.pdf
"""
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers
from gate.net import net


class resnet_cifar(net.Net):

    def __init__(self):
        # regularization
        self.weight_decay = 0.0002
        # for bn
        self.batch_norm_epsilon = 1e-5  # 0.001
        self.batch_norm_decay = 0.997
        self.batch_norm_scale = True
        # config
        self.use_bottleneck = False
        self.num_residual_units = 5

    def arg_scope(self):
        weight_decay = self.weight_decay
        batch_norm_decay = self.batch_norm_decay
        batch_norm_epsilon = self.batch_norm_epsilon
        batch_norm_scale = self.batch_norm_scale

        batch_norm_params = {
            'decay': batch_norm_decay,
            'epsilon': batch_norm_epsilon,
            'scale': batch_norm_scale,
            'updates_collections': None,
            'zero_debias_moving_mean': True
        }

        with arg_scope([layers.conv2d],
                       weights_regularizer=layers.l2_regularizer(weight_decay),
                       weights_initializer=tf.truncated_normal_initializer(
                           stddev=0.05),
                       biases_initializer=None,
                       activation_fn=tf.nn.relu,
                       normalizer_fn=layers.batch_norm,
                       normalizer_params=batch_norm_params,
                       padding='SAME'):
            with arg_scope([layers.batch_norm], **batch_norm_params):
                with arg_scope([layers.max_pool2d, layers.avg_pool2d], padding='SAME') as arg_sc:
                    return arg_sc

    def model(self, images, num_classes, is_training):
        end_points = {}
        with arg_scope([layers.batch_norm], is_training=is_training):
            x = layers.conv2d(images, 16, [3, 3], 1)

            strides = [1, 2, 2]
            activate_before_residual = [True, False, False]

            if self.use_bottleneck:
                res_func = self.bottleneck_residual
                filters = [16, 64, 128, 256]
            else:
                res_func = self.residual
                filters = [16, 16, 32, 64]
            # Uncomment the following codes to use w28-10 wide residual network.
            # It is more memory efficient than very deep residual network and has
            # comparably good performance.
            # https://arxiv.org/pdf/1605.07146v1.pdf
            # filters = [16, 160, 320, 640]
            # Update num_residual_units to 9

            with tf.variable_scope('unit_1_0'):
                x = res_func(x, filters[0], filters[1], strides[0],
                             activate_before_residual[0])
                for i in range(self.num_residual_units):
                    with tf.variable_scope('unit_1_%d' % i):
                        x = res_func(x, filters[1], filters[1], 1, False)

            with tf.variable_scope('unit_2_0'):
                x = res_func(x, filters[1], filters[2], strides[1],
                             activate_before_residual[1])
                for i in range(self.num_residual_units):
                    with tf.variable_scope('unit_1_%d' % i):
                        x = res_func(x, filters[2], filters[2], 1, False)

            with tf.variable_scope('unit_3_0'):
                x = res_func(x, filters[2], filters[3], strides[2],
                             activate_before_residual[2])
                for i in range(self.num_residual_units):
                    with tf.variable_scope('unit_1_%d' % i):
                        x = res_func(x, filters[3], filters[3], 1, False)

            with tf.variable_scope('unit_last'):
                x = tf.nn.relu(layers.batch_norm(x))
                x = layers.avg_pool2d(x, [8, 8], 1, padding='VALID')
                logits = layers.fully_connected(
                    x, num_classes,
                    biases_initializer=tf.zeros_initializer(),
                    weights_initializer=tf.truncated_normal_initializer(
                        stddev=0.01),
                    weights_regularizer=None,
                    activation_fn=None,
                    scope='logits')

                end_points['logits'] = end_points
                return logits, end_points

    def residual(self, x, in_filter, out_filter, stride,
                 activate_before_residual=False):
        """ Residual unit with 2 sub layers. """
        if activate_before_residual:
            x = tf.nn.relu(layers.batch_norm(x))
            orig_x = x
        else:
            orig_x = x
            x = tf.nn.relu(layers.batch_norm(x))

        x = layers.conv2d(x, out_filter, [3, 3], stride)
        x = layers.conv2d(x, out_filter, [3, 3], 1)

        if in_filter != out_filter:
            orig_x = layers.avg_pool2d(orig_x, 2, 2)
            orig_x = layers.conv2d(orig_x, out_filter, [1, 1], 1)
        x += orig_x

        return x

    def bottleneck_residual(self, x, in_filter, out_filter, stride,
                            activate_before_residual=False):
        """ Bottleneck residual unit with 3 sub layers. """
        if activate_before_residual:
            x = tf.nn.relu(layers.batch_norm(x))
            orig_x = x
        else:
            orig_x = x
            x = tf.nn.relu(layers.batch_norm(x))

        x = layers.conv2d(x, out_filter / 4, [1, 1], stride)
        x = layers.conv2d(x, out_filter / 4, [3, 3], 1)
        x = layers.conv2d(x, out_filter, [1, 1], 1)

        if in_filter != out_filter:
            orig_x = layers.conv2d(orig_x, out_filter, [1, 1], stride)
        x += orig_x

        return x
