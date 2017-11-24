# -*- coding: utf-8 -*-
""" Public base for CNN task
    updated: 2017/11/24
"""
import time
import tensorflow as tf
import numpy as np

from core.utils.logger import logger
from core.solver.snapshot import Snapshot
from core.solver.summary import Summary


class Context():
  """ A common class offers context manage
  """

  def __init__(self, config):
    self.config = config
    self.phase = config.phase
    self.data = config.data
    self.summary = Summary(config)
    self.snapshot = Snapshot(config)

  def _enter_(self, phase):
    self.prephase = self.phase
    self.phase = phase
    self.config.set_phase(phase)
    self.data = self.config.data

  def _exit_(self):
    self.phase = self.prephase
    self.config.set_phase(self.phase)
    self.data = self.config.data


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

  def before_run(self, run_context):
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

    if (cur_iter - 1) % self.config.val_invl == 0 and cur_iter > 10:
      if self.func_val is not None:
        with tf.Graph().as_default():
          self.func_val()

    if (cur_iter - 1) % self.config.test_invl == 0 and cur_iter > 10:
      if self.func_test is not None:
        with tf.Graph().as_default():
          self.func_test()

    if cur_iter == self.config.max_iter:
      logger.sys('Has achieved the maximum iterations, \
          the system will terminate.')
      exit(0)


class QueueContext():
  """ For managing the data reader queue.
  """

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
