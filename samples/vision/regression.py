# -*- coding: utf-8 -*-
"""
DECTRION FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2018/4/21

--------------------------------------------------------

Regrssion for CNN

"""

import tensorflow as tf
from gate import context
from gate.net.net_factory import net_graph
from gate.data.data_factory import load_data
from gate.solver import updater
from gate.layer import l2
from gate.utils import variable
from gate.utils import filesystem
from gate.utils import string
from gate.utils.logger import logger


class Regression(context.Context):

  def __init__(self, config):
    context.Context.__init__(self, config)

  def _net(self, data):
    return net_graph(data, self.config.net[0], self.phase)

  def _loss(self, logit, label):
    loss = l2.loss(logit, label, self.config)
    return loss

  def _error(self, logit, label):
    mae, rmse = l2.error(logit, label, self.config)
    return mae, rmse

  def train(self):
    self._enter_('train')
    with tf.Graph().as_default() as graph:
      # load data
      image, label, path = load_data(self.config)
      # load net
      logit, end_points = self._net(image)
      # compute loss
      loss = self._loss(logit, label)
      # compute error
      mae, rmse = self._error(logit, label)
      # update gradients
      global_step = tf.train.create_global_step()
      train_op = updater.default(self.config, loss, global_step)
      # add hooks
      self.add_hook(self.snapshot.init())
      self.add_hook(self.summary.init())
      self.add_hook(context.Running_Hook(
          config=self.config.log,
          step=global_step,
          keys=['loss', 'mae', 'rmse'],
          values=[loss, mae, rmse],
          func_test=self.test,
          func_val=None))

      saver = tf.train.Saver(var_list=variable.all())
      with context.DefaultSession(self.hooks) as sess:
        self.snapshot.restore(sess, saver)
        while not sess.should_stop():
          sess.run(train_op)

    self._exit_()

  def test(self):
    self._enter_('test')
    # create a folder to save
    test_dir = filesystem.mkdir(self.config.output_dir + '/test/')
    # get data
    image, label, path = load_data(self.config)
    # get net
    logit, end_points = self._net(image)
    # get loss
    loss = self._loss(logit, label)
    # get logit
    mae, rmse = self._error(logit, label)
    # output to file
    info = string.concat(self.batchsize, [path, label, logit])
    mean_loss, mean_mae, mean_rmse = 0, 0, 0
    saver = tf.train.Saver()

    # running test
    with context.DefaultSession() as sess:
      global_step = self.snapshot.restore(sess, saver)
      with open(test_dir + '%s.txt' % global_step, 'wb') as fw:
        with context.QueueContext(sess):
          for _ in range(self.num_batch):
            _loss, _mae, _rmse, _info = sess.run([loss, mae, rmse, info])
            mean_loss += _loss
            mean_mae += _mae
            mean_rmse += _rmse
            [fw.write(_line + b'\r\n') for _line in _info]

      # statistic
      mean_loss = 1.0 * mean_loss / self.num_batch
      mean_mae = 1.0 * mean_mae / self.num_batch
      mean_rmse = 1.0 * mean_rmse / self.num_batch

      # display results on screen
      keys = ['total sample', 'num batch', 'loss', 'mae', 'rmse']
      vals = [self.total_num, self.num_batch, mean_loss, mean_mae, mean_rmse]
      logger.test(logger.iters(int(global_step), keys, vals))

      # write to summary
      self.summary.adds(global_step=global_step,
                        tags=['test/mae', 'test/rmse', 'test/loss'],
                        values=[mean_mae, mean_rmse, mean_loss])

      self._exit_()
      return mean_rmse
