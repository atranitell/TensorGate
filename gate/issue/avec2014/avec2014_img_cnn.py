# -*- coding: utf-8 -*-
"""
DECTRION FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2018/2/25

--------------------------------------------------------

FOR AVEC2014

"""

import tensorflow as tf
from gate import context
from gate.net.factory import get_net
from gate.data.factory import get_data
from gate.solver import updater
from gate.layer import l2
from gate.util import variable
from gate.util import filesystem
from gate.util import string
from gate.util.logger import logger
from gate.issue.avec2014.utils import get_accurate_from_file


class AVEC2014_IMG_CNN(context.Context):
  """ Use normal cnn training single image.
  """

  def __init__(self, config):
    context.Context.__init__(self, config)

  def _net(self, data):
    return get_net(data, self.config.net[0], self.phase)

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
      image, label, path = get_data(self.config)
      # load net
      logit, _ = self._net(image)
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
    image, label, path = get_data(self.config)
    # get net
    logit, _ = self._net(image)
    # output to file
    info = string.concat(self.batchsize, [path, label, logit*self.data.span])
    saver = tf.train.Saver()

    # running test
    with context.DefaultSession() as sess:
      global_step = self.snapshot.restore(sess, saver)
      result_path = test_dir + '%s.txt' % global_step
      with open(result_path, 'wb') as fw:
        with context.QueueContext(sess):
          for _ in range(self.num_batch):
            _info = sess.run(info)
            [fw.write(_line + b'\r\n') for _line in _info]

      # display results on screen
      _mae, _rmse = get_accurate_from_file(result_path)
      keys = ['total sample', 'num batch', 'video_mae', 'video_rmse']
      vals = [self.total_num, self.num_batch, _mae, _rmse]
      logger.test(logger.iters(int(global_step), keys, vals))

      # write to summary
      self.summary.adds(global_step=global_step,
                        tags=['test/video_mae', 'test/video_rmse'],
                        values=[_mae, _rmse])

      self._exit_()
      return _rmse
