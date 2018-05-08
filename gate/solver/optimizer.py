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
"""Optimizer"""

import tensorflow as tf
from gate.utils.logger import logger


def configure_optimizer(config, learning_rate):
  """Configures the optimizer used for training.

  Args:
    config: config.train.optimizer
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if opt.optimizer is not recognized.

  """
  logger.info('Routine will use %s optimizer.' % config.name)

  if config.name == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=config.rho,
        epsilon=config.epsilon)

  elif config.name == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=config.initial_accumulator_value)

  elif config.name == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=config.beta1,
        beta2=config.beta2,
        epsilon=config.epsilon)

  elif config.name == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=config.learning_rate_power,
        initial_accumulator_value=config.initial_accumulator_value,
        l1_regularization_strength=config.l1,
        l2_regularization_strength=config.l2)

  elif config.name == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=config.momentum,
        name='Momentum')

  elif config.name == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=config.decay,
        momentum=config.momentum,
        epsilon=config.epsilon)

  elif config.name == 'proximal':
    optimizer = tf.train.ProximalGradientDescentOptimizer(
        learning_rate,
        l1_regularization_strength=config.l1_regularization_strength,
        l2_regularization_strength=config.l2_regularization_strength)

  elif config.name == 'proximal_adagrad':
    optimizer = tf.train.ProximalAdagradOptimizer(
        learning_rate,
        initial_accumulator_value=config.initial_accumulator_value,
        l1_regularization_strength=config.l1_regularization_strength,
        l2_regularization_strength=config.l2_regularization_strength)

  elif config.name == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

  else:
    raise ValueError('Optimizer [%s] was not recognized' % config.name)

  return optimizer
