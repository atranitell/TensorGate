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
"""CONTEXT for all issues"""

import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from gate.env import env
from gate.utils import filesystem
from gate.utils import string
from gate.utils import variable
from gate.utils.logger import logger
from gate.solver import snapshot
from gate.solver import summary


class Context():
  """ A common class offers context managers, tasks should inherit this class.
  """

  def __init__(self, config):
    """ the self.phase and self.data used for store for switching task
      like 'train' to 'test'.
    """
    self.config = config
    self.phase = None
    self.data = None

    """ initialize logger """
    pid = datetime.strftime(datetime.now(), '%y%m%d%H%M%S')
    # if output_dir is None, to make a new dir to save model
    # else we use the value of output_dir as workspace
    filename = string.join_dots(self.config.name, self.config.target, pid)
    if self.config.output_dir is None:
      self.config.output_dir = filesystem.mkdir(env._OUTPUT + filename)
    logger.init(filename, self.config.output_dir)
    logger.info('Initilized logger successful.')
    logger.info('Current model in %s' % self.config.output_dir)

    """ initialize auxiliary information """
    self.hooks = []
    self.summary = summary.Summary(self.config)
    self.snapshot = snapshot.Snapshot(self.config)

    # print config information
    string.print_members(config)

  def _enter_(self, phase):
    self.prephase = self.phase
    self.phase = phase
    self.config.set_phase(phase)
    self.data = self.config.data

  def _exit_(self):
    if self.prephase is None:
      return
    self.phase = self.prephase
    self.config.set_phase(self.phase)
    self.data = self.config.data

  def add_hook(self, hook):
    self.hooks.append(hook)

  @property
  def is_training(self):
    return True if self.phase == 'train' else False

  @property
  def batchsize(self):
    return self.config.data.batchsize

  @property
  def total_num(self):
    return self.config.data.total_num

  @property
  def iter_per_epoch(self):
    return int(self.config.data.total_num / self.config.data.batchsize)

  @property
  def num_batch(self):
    return int(self.data.total_num / self.data.batchsize)


class Running_Hook(tf.train.SessionRunHook):
  """ Running Hooks for training showing information """

  def __init__(self, config, step, keys, values,
               func_val=None, func_test=None):
    """ Running session for common application.
        Default values[0] is iteration
        config: config.log
    """
    self.duration = 0
    self.values = values
    self.mean_values = np.zeros(len(self.values) + 1)
    self.keys = keys + ['time']
    self.step = step
    self.config = config
    # call
    self.func_val = func_val
    self.func_test = func_test

  def begin(self):
    pass
    # display variables
    # variable.print_trainable_list()
    # variables.print_global_list()

  def before_run(self, run_context):
    # feed monitor values
    self.start_time = time.time()
    return tf.train.SessionRunArgs([self.step] + self.values)

  def after_run(self, run_context, run_values):
    cur_iter = run_values.results[0] - 1
    self.mean_values[:-1] += run_values.results[1:]
    self.mean_values[-1] += (time.time() - self.start_time) * 1000

    if cur_iter % self.config.print_invl == 0 and cur_iter != 0:
      self.mean_values /= self.config.print_invl
      logger.train(logger.iters(cur_iter, self.keys, self.mean_values))
      np.zeros_like(self.mean_values)

    if (cur_iter) > 10 and (cur_iter - 1) % self.config.val_invl == 0:
      if self.func_val is not None:
        with tf.Graph().as_default():
          self.func_val()

    if (cur_iter) > 10 and (cur_iter - 1) % self.config.test_invl == 0:
      if self.func_test is not None:
        with tf.Graph().as_default():
          self.func_test()

    if cur_iter == self.config.max_iter:
      logger.sys('Achieved the maximum iterations, the system will terminate.')
      exit(0)


class QueueContext():
  """For managing the data reader queue."""

  def __init__(self, sess):
    self.sess = sess

  def __enter__(self):
    self.coord = tf.train.Coordinator()
    self.threads = []
    for queuerunner in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
      self.threads.extend(queuerunner.create_threads(
          self.sess, coord=self.coord, daemon=True, start=True))

  def __exit__(self, *unused):
    self.coord.request_stop()
    self.coord.join(self.threads, stop_grace_period_secs=10)


class DefaultSession():
  """Default session for custom"""

  def __init__(self, hooks=None):
    self.hooks = hooks
    self.sess = None

  def __enter__(self):
    """ there, we set all issue to configure gpu memory with auto growth
      however, when train + test, the memory will increase.
    ----
      test presents that the performance has no benefits.
    """
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    if self.hooks is not None:
      self.sess = tf.train.MonitoredTrainingSession(
          hooks=self.hooks,
          save_checkpoint_secs=None,
          save_summaries_steps=None)
      return self.sess
    else:
      self.sess = tf.Session()
      return self.sess

  def __exit__(self, *unused):
    self.sess.close()
