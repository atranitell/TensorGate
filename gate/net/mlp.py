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
"""Contains a variant of the MLP model definition."""

import tensorflow as tf
from gate.net import net

slim = tf.contrib.slim


class MLP(net.Net):

    def arguments_scope(self):
        return mlp_arg_scope(self.weight_decay)

    def model(self, inputs, num_classes, is_training):
        return mlp(inputs, num_classes, is_training, self.dropout)


def mlp(inputs, num_classes, is_training=False,
        dropout_keep_prob=0.5,
        scope='MLP'):
    """Creates a variant of the MLP model.
        inputs is (batchsize, N) dim.
    """
    end_points = {}

    with tf.variable_scope(scope, 'MLP', [inputs, num_classes]):
        net = slim.fully_connected(inputs, 1024, scope='fc1')
        net = slim.dropout(
            net, dropout_keep_prob, is_training=is_training, scope='dropout1')
        logits = slim.fully_connected(
            net, num_classes, activation_fn=None, scope='fc2')

    end_points['Logits'] = logits

    return logits, end_points


def mlp_arg_scope(weight_decay=0.0):
    """Defines the default lenet argument scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.

    Returns:
      An `arg_scope` to use for the inception v3 model.
    """
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
            activation_fn=tf.nn.relu) as sc:
        return sc
