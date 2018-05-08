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
"""L2 LOSS OPS"""

import tensorflow as tf
from gate.env import env


def loss(logits, labels, config):
  """L2 loss: |labels - logits|^2

  Args:
    scale: true for scale labels from [0, span] to [0, 1]

  Returns:
    _losses: a scalr with float32
  """
  with tf.name_scope('l2_loss'):
    _logits = tf.to_float(tf.reshape(logits, [config.data.batchsize, 1]))
    _labels = tf.to_float(tf.reshape(labels, [config.data.batchsize, 1]))
    if config.data.scale:
      _labels = tf.divide(_labels, config.data.span)
    _losses = tf.nn.l2_loss([_labels - _logits], name='loss')
    if env._SUMMARY_SCALAR:
      tf.summary.scalar('losses', _losses)
    return _losses


def error(logits, labels, config):
  """Return mae and rmse value

  Args:
    logits: float32
    labels: float32 and identical scale with logits

  Returns:
    mae: mean absolute error
    rmse: root of mean square error
  """
  with tf.name_scope('error'):
    _logits = tf.to_float(tf.reshape(logits, [config.data.batchsize, 1]))
    _labels = tf.to_float(tf.reshape(labels, [config.data.batchsize, 1]))
    _span = 1 if not config.data.scale else config.data.span
    _error = _logits*_span - _labels
    err_mae = tf.reduce_mean(tf.abs(_error, name='error_mae'))
    err_rmse = tf.sqrt(tf.reduce_mean(tf.square(_error, name='error_mse')))
    if env._SUMMARY_SCALAR:
      tf.summary.scalar('mae', err_mae)
      tf.summary.scalar('rmse', err_rmse)
    return err_mae, err_rmse
