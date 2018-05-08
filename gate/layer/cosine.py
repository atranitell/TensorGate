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
"""Loss function for cosine distance"""

import tensorflow as tf
from gate.env import env


def loss(x, y, labels, batch_size, is_training=True):
  """Cosine loss: 1-<x , y>/(norm(x)*norm(y))*label

  Args:
    labels: +1 / -1

  Returns:
    losses: a scalr with float32
    loss: a batchsize of loss for per sample
  """
  with tf.name_scope('cosine_loss'):
    norm_x = tf.reshape(tf.norm(x, axis=1), [batch_size, 1])
    norm_y = tf.reshape(tf.norm(y, axis=1), [batch_size, 1])

    x1 = tf.expand_dims(x, axis=2)
    x1 = tf.transpose(x1, perm=[0, 2, 1])
    y1 = tf.expand_dims(y, axis=2)

    loss = tf.reshape(
        tf.matmul(x1, y1), [batch_size, 1]) / (norm_x * norm_y)

    if is_training:
      label = tf.reshape(labels, [batch_size, 1])
      loss = loss * tf.to_float(label)

    losses = 1.0 - tf.reduce_mean(loss)
    loss = 1 - loss

    if env._SUMMARY_SCALAR:
      tf.summary.scalar('loss', losses)

    return losses, loss
