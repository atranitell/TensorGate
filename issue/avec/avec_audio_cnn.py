# -*- coding: utf-8 -*-
""" regression task for audio
    updated: 2018/02/05
"""
import tensorflow as tf
from core.data.factory import loads
from core.network.factory import network
from core.loss import l2
from core.solver import updater
from core.solver import context
from core.utils.logger import logger
from core.utils.string import string
from core.utils.variables import variables
from core.utils.filesystem import filesystem

from issue.avec.avec_utils import get_accurate_from_file
import numpy as np


class AVEC_AUDIO_CNN(context.Context):
  """
  Experiment1:
    - 4-hierarchy better than 3-hierarchy
    - smaller frame length better than larger
    - combine multi-small frame ?
  """

  def __init__(self, config):
    context.Context.__init__(self, config)

  def _net(self, X):
    X = tf.reshape(
        X, [self.batchsize,
            self.data.configs[0].frame_length *
            self.config.data.configs[0].frame_num, 1])
    logit, net = network(X, self.config, self.phase)
    return logit, net

  def _loss(self, logit, label):
    loss, logits, labels = l2.get_loss(logit, label, self.config)
    mae, rmse = l2.get_error(logits, labels, self.config)
    return loss, mae, rmse

  def train(self):
    """
    """
    # set phase
    self._enter_('train')

    # get data pipeline
    data, label, path = loads(self.config)

    # get network
    logit, net = self._net(data)
    # get loss
    loss, mae, rmse = self._loss(logit, label)

    # update
    global_step = tf.train.create_global_step()
    train_op = updater.default(self.config, loss, global_step)

    # update at the same time
    saver = tf.train.Saver(var_list=variables.all())

    # hooks
    self.add_hook(self.snapshot.init())
    self.add_hook(self.summary.init())
    self.add_hook(context.Running_Hook(
        config=self.config.log,
        step=global_step,
        keys=['loss', 'mae', 'rmse'],
        values=[loss, mae, rmse],
        func_test=self.test,
        func_val=self.val))

    with context.DefaultSession(self.hooks) as sess:
      self.snapshot.restore(sess, saver)
      while not sess.should_stop():
        sess.run(train_op)

  def test(self):
    """
    """
    # save current context
    self._enter_('test')

    # create a folder to save
    test_dir = filesystem.mkdir(self.config.output_dir + '/test/')

    # get data pipeline
    data, label, path = loads(self.config)
    # total_num
    total_num = self.data.total_num
    batchsize = self.data.batchsize

    # get network
    logit, net = self._net(data)
    # get loss
    loss, mae, rmse = self._loss(logit, label)

    # get saver
    saver = tf.train.Saver()
    with context.DefaultSession() as sess:
      # get latest checkpoint
      global_step = self.snapshot.restore(sess, saver)
      info = string.concat(batchsize, [path, label, logit * self.data.range])
      filename = test_dir + '%s.txt' % global_step
      # start to run
      with open(filename, 'wb') as fw:
        with context.QueueContext(sess):
          # Initial some variables
          mean_loss, num_iter = 0., int(total_num / batchsize)

          for _ in range(num_iter):
            # running session to acuqire value
            _loss, _info = sess.run([loss, info])
            mean_loss += _loss
            # save tensor info to text file
            [fw.write(_line + b'\r\n') for _line in _info]

          # statistic
          mean_loss = 1.0 * mean_loss / num_iter

      # display results on screen
      _mae, _rmse = get_accurate_from_file(filename)
      keys = ['total sample', 'num batch', 'loss', 'video_mae', 'video_rmse']
      vals = [total_num, num_iter, mean_loss, _mae, _rmse]
      logger.test(logger.iters(int(global_step), keys, vals))

      # write to summary
      self.summary.adds(
          global_step=global_step,
          tags=['test/loss', 'test/video_mae', 'test/video_rmse'],
          values=[mean_loss, _mae, _rmse])

      self._exit_()
      return _mae

  def val(self):
    """
    """
    # save current context
    self._enter_('val')

    # create a folder to save
    val_dir = filesystem.mkdir(self.config.output_dir + '/val/')

    # get data pipeline
    data, label, path = loads(self.config)
    # total_num
    total_num = self.data.total_num
    batchsize = self.data.batchsize

    # get network
    logit, net = self._net(data)
    # get loss
    loss, mae, rmse = self._loss(logit, label)

    # get saver
    saver = tf.train.Saver()
    with context.DefaultSession() as sess:
      global_step = self.snapshot.restore(sess, saver)
      info = string.concat(batchsize, [path, label, logit * self.data.range])
      filename = val_dir + '%s.txt' % global_step
      # start to run
      with open(filename, 'wb') as fw:
        with context.QueueContext(sess):
          # Initial some variables
          mean_loss, num_iter = 0., int(total_num / batchsize)

          for _ in range(num_iter):
            # running session to acuqire value
            _loss, _info = sess.run([loss, info])
            mean_loss += _loss
            # save tensor info to text file
            [fw.write(_line + b'\r\n') for _line in _info]

          # statistic
          mean_loss = 1.0 * mean_loss / num_iter

      # display results on screen
      _mae, _rmse = get_accurate_from_file(filename)
      keys = ['total sample', 'num batch', 'loss', 'video_mae', 'video_rmse']
      vals = [total_num, num_iter, mean_loss, _mae, _rmse]
      logger.val(logger.iters(int(global_step), keys, vals))

      # write to summary
      self.summary.adds(
          global_step=global_step,
          tags=['val/loss', 'val/video_mae', 'val/video_rmse'],
          values=[mean_loss, _mae, _rmse])

      self._exit_()
      return _mae

  def heatmap(self):
    """
    """
    # save current context
    self._enter_('test')

    # create a folder to save
    test_dir = filesystem.mkdir(self.config.output_dir + '/test_heatmap/')
    data, label, path = loads(self.config)
    logit, net = self._net(data)
    loss, mae, rmse = self._loss(logit, label)
    out_logit = logit * self.data.range
    saver = tf.train.Saver()

    data = tf.reshape(data, [self.batchsize, -1])
    d_fc = net['gap_conv']
    w_fc = variables.select_vars('sen1_ma_1111/logits/weights')[0]

    with context.DefaultSession() as sess:
      global_step = self.snapshot.restore(sess, saver)
      with context.QueueContext(sess):
        n_dx = []
        n_dfc = []
        n_wfc = []
        n_path = []
        n_label = []
        n_pred = []
        for i in range(int(self.data.total_num / self.data.batchsize)):
          _x, _dfc, _wfc, _p, _l, _pred = sess.run([data, d_fc, w_fc, path, label, out_logit])
          n_dx.append(_x)
          n_dfc.append(_dfc)
          n_wfc.append(_wfc)
          n_path.append(_p)
          n_label.append(_l)
          n_pred.append(_pred)
          if i % 10 == 0:
            print(i * 50)

        np.save('data.npy', np.array(n_dx))
        np.save('dfc.npy', np.array(n_dfc))
        np.save('wfc.npy', np.array(n_wfc))
        np.save('path.npy', np.array(n_path))
        np.save('label.npy', np.array(n_label))
        np.save('pred.npy', np.array(n_pred))
        # exit(1)

      self._exit_()
      return 0
