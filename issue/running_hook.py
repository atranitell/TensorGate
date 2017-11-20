# -*- coding: utf-8 -*-
""" Running Hooks for training showing information
    updated: 2017/11/19
"""

import time
import tensorflow as tf
import numpy as np
from core.utils.logger import logger


class Running_Hook(tf.train.SessionRunHook):

  def __init__(self, config, step, keys, values,
               func_val=None, func_test=None):
    """ Running session for common application.
        Default values[0] is iteration
        config: config[config['task']]['log']
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
    self.mean_values[-1] += time.time() - self.start_time

    if cur_iter % self.config['display_iter'] == 0 and cur_iter != 0:
      self.mean_values /= self.config['display_iter']
      logger.train(logger.iters(cur_iter, self.keys, self.mean_values))
      np.zeros_like(self.mean_values)

    if (cur_iter - 1) % self.config['val_iter'] == 0 and cur_iter > 10:
      if self.func_val is not None:
        with tf.Graph().as_default():
          self.func_val()

    if (cur_iter - 1) % self.config['test_iter'] == 0 and cur_iter > 10:
      if self.func_test is not None:
        with tf.Graph().as_default():
          self.func_test()

    if cur_iter == self.config['max_iter']:
      logger.sys('Has achieved the maximum iterations, \
          the system will terminate.')
      exit(0)
