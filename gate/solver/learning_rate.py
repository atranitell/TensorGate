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
"""Learning Rate"""

import tensorflow as tf


def decay_steps(total_num, batchsize, decay_epochs):
  return int(total_num / batchsize * decay_epochs)


def configure_lr(config, global_step, batchsize, total_num):
  """Configures the learning rate.

  Args:
    config: config.train.lr
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the learning rate.
    
  """
  if config.name == 'exponential':
    return tf.train.exponential_decay(
        config.learning_rate,
        global_step,
        decay_steps(total_num, batchsize, config.decay_epochs),
        config.decay_factor,
        staircase=False,
        name='exponential_decay_learning_rate')

  elif config.name == 'fixed':
    return tf.constant(config.learning_rate, name='fixed_learning_rate')

  elif config.name == 'vstep':
    global_step = tf.to_int32(global_step)
    return tf.train.piecewise_constant(
        global_step,
        config.boundaries,
        config.values,
        name='vstep_decay_learning_rate')

  elif config.name == 'polynomial':
    return tf.train.polynomial_decay(
        config.learning_rate,
        global_step,
        decay_steps(total_num, batchsize, config.decay_epochs),
        config.end_learning_rate,
        power=1.0,
        cycle=False,
        name='polynomial_decay_learning_rate')

  elif config.name == 'natural_exp':
    return tf.train.natural_exp_decay(
        config.learning_rate,
        global_step,
        decay_steps(total_num, batchsize, config.decay_epochs),
        config.decay_rate,
        staircase=True,
        name='natural_exp_decay_learning_rate')

  elif config.name == 'cosine':
    return tf.train.cosine_decay(
        config.learning_rate,
        global_step,
        decay_steps(total_num, batchsize, config.decay_epochs),
        name="cosine_decay_learning_rate")

  elif config.name == 'linear_cosine':
    return tf.train.linear_cosine_decay(
        config.learning_rate,
        global_step,
        decay_steps(total_num, batchsize, config.decay_epochs),
        name="linear_cosine_decay_learning_rate")

  elif config.name == 'noisy_linear_cosine':
    return tf.train.noisy_linear_cosine_decay(
        config.learning_rate,
        global_step,
        decay_steps(total_num, batchsize, config.decay_epochs),
        name="noisy_linear_cosine_decay_learning_rate")

  elif config.name == 'inverse_time':
    return tf.train.inverse_time_decay(
        config.learning_rate,
        global_step,
        decay_steps(total_num, batchsize, config.decay_epochs),
        config.decay_rate,
        name="inverse_time_decay_learning_rate")

  else:
    error = 'learning rate type [ % s] was not recognized' % (config.name)
    raise ValueError(error)
