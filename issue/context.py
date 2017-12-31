# -*- coding: utf-8 -*-
""" Public base for All task
    updated: 2017/11/24
"""
import time
import shutil
import numpy as np
import tensorflow as tf
from datetime import datetime
from core.utils.logger import logger
from core.solver.snapshot import Snapshot
from core.solver.summary import Summary
from core.solver import variables
from core.utils.string import print_members
from core.utils import filesystem
from tools import checkpoint

OUTPUTS = filesystem.mkdir('../_outputs/')


class Context():
  """ A common class offers context manage
  """

  def __init__(self, config):
    self.config = config
    self._initialize()
    #-- inner variables --#
    self.phase = None
    self.data = None
    self.summary = Summary(self.config)
    self.snapshot = Snapshot(self.config)
    print_members(self.config)

  def _initialize(self):
    pid = datetime.strftime(datetime.now(), '%y%m%d%H%M%S')
    # 1. setting output dir
    if self.config.output_dir is None:
      self.config.output_dir = filesystem.mkdir(
          OUTPUTS + self.config.name + '.' + self.config.target + '.' + pid)

    # 2. setting logger location
    logger.init(self.config.name + '.' + pid, self.config.output_dir)
    logger.info('Initilized logger successful.')
    logger.info('Current model in %s' % self.config.output_dir)

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

  @property
  def is_train(self):
    return True if self.phase == 'train' else False

  @property
  def batchsize(self):
    return self.config.data.batchsize

  @property
  def epoch_iter(self):
    return int(self.config.data.total_num / self.config.data.batchsize)

  def pipline(self):
    """ traverse all checkpoints """
    ckpt_path = self.config.output_dir + '/checkpoint'
    ckpt_path_bk = ckpt_path + '.bk'
    shutil.copy(ckpt_path, ckpt_path_bk)
    model_list = checkpoint.get_checkpoint_model_items(ckpt_path)
    for item in model_list[1:]:
      model_list[0] = item
      checkpoint.write_checkpoint_model_items(ckpt_path, model_list)
      self.test()
    shutil.copy(ckpt_path_bk, ckpt_path)


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
    # display variables
    variables.print_trainable_list()
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
