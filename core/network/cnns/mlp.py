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
import tensorflow.contrib.layers as layers


def mlp(inputs, num_classes, scope='MLP'):
  """Creates a variant of the MLP model.
      inputs is (batchsize, N) dim.
  """
  end_points = {}
  with tf.variable_scope(scope, 'MLP'):
    logits = layers.fully_connected(
        inputs,
        512,
        biases_initializer=tf.constant_initializer(0.0),
        weights_regularizer=layers.l2_regularizer(0.0001),
        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
        activation_fn=None)
  return logits, end_points
