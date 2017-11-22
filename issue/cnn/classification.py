# -*- coding: utf-8 -*-
""" classification task for image
    updated: 2017/11/19
"""

import os
import tensorflow as tf
from core.loss import softmax
from core.database.dataset import Dataset
from core.network.cnn import network
from core.solver.updater import Updater
from core.solver.snapshot import Snapshot
from core.solver.summary import Summary

from core.utils import string
from core.utils import filesystem
from core.utils.logger import logger
from core.utils.context import QueueContext

from issue.running_hook import Running_Hook


class classification():

  def __init__(self, config):
    self.config = config
    self.summary = Summary(self.config['log'], config['output_dir'])
    self.snapshot = Snapshot(self.config['log'], config['output_dir'])
    # current work env
    self.taskcfg = None

  def _enter_(self, phase):
    """ task enter
    """
    self.pre_taskcfg = self.taskcfg
    self.taskcfg = self.config[phase]
    self.datacfg = self.taskcfg['data']

  def _exit_(self):
    """ task exit
    """
    self.taskcfg = self.pre_taskcfg
    self.datacfg = self.taskcfg['data']

  def _net(self, data, phase):
    logit, net = network(data, self.config, phase)
    return logit, net

  def _loss(self, logit, label):
    # get loss
    loss, logit = softmax.get_loss(
        logit, label, self.taskcfg['data']['num_classes'],
        self.taskcfg['data']['batchsize'])

    # get error
    error, pred = softmax.get_error(logit, label)
    return loss, error, pred

  def train(self):
    """
    """
    # set phase
    self._enter_('train')

    # get data pipeline
    data, label, path = Dataset(self.datacfg, 'train').loads()
    # get network
    logit, net = self._net(data, 'train')
    # get loss
    loss, error, pred = self._loss(logit, label)

    # update
    with tf.name_scope('updater'):
      global_step = tf.train.create_global_step()
      updater = Updater(global_step)
      updater.init_default_updater(self.taskcfg, loss)
      train_op = updater.get_train_op()
      restore_saver = updater.get_variables_saver()

    # hooks
    snapshot_hook = self.snapshot.init()
    summary_hook = self.summary.init()
    running_hook = Running_Hook(
        config=self.config['log'],
        step=global_step,
        keys=['loss', 'error'],
        values=[loss, error],
        func_test=self.test,
        func_val=None)

    # monitor session
    with tf.train.MonitoredTrainingSession(
            hooks=[running_hook, snapshot_hook, summary_hook,
                   tf.train.NanTensorHook(loss)],
            save_checkpoint_secs=None,
            save_summaries_steps=None) as sess:

      # restore model
      if 'restore' in self.taskcfg and self.taskcfg['restore']:
        self.snapshot.restore(sess, restore_saver)

      # running
      while not sess.should_stop():
        sess.run(train_op)

  def test(self):
    """
    """
    # save current context
    self._enter_('test')

    # create a folder to save
    test_dir = filesystem.mkdir(self.config['output_dir'] + '/test/')

    # get data pipeline
    data, label, path = Dataset(self.datacfg, 'test').loads()
    # alias
    batchsize = self.datacfg['batchsize']
    total_num = self.datacfg['total_num']
    # get network
    logit, net = self._net(data, 'test')
    # get loss
    loss, error, pred = self._loss(logit, label)

    # get saver
    saver = tf.train.Saver(name='restore_all')
    with tf.Session() as sess:
      # get latest checkpoint
      global_step = self.snapshot.restore(sess, saver)
      # Initial some variables
      num_iter = int(total_num / batchsize)
      mean_err, mean_loss = 0, 0
      # output to file
      info = string.concat_str_in_tab(batchsize, [path, label, pred])
      with open(test_dir + '%s.txt' % global_step, 'wb') as fw:
        with QueueContext(sess):
          for _ in range(num_iter):
            # running session to acuqire value
            _loss, _err, _info = sess.run([loss, error, info])
            mean_loss += _loss
            mean_err += _err
            # save tensor info to text file
            [fw.write(_line + b'\r\n') for _line in _info]

      # statistic
      mean_loss = 1.0 * mean_loss / num_iter
      mean_err = 1.0 * mean_err / num_iter

      # display results on screen
      keys = ['total sample', 'num batch', 'loss', 'error']
      vals = [total_num, num_iter, mean_loss, mean_err]
      logger.test(logger.iters(int(global_step), keys, vals))

      # write to summary
      self.summary.adds(global_step=global_step,
                        tags=['test/error', 'test/loss'],
                        values=[mean_err, mean_loss])

      self._exit_()
      return mean_err

  def val(self):
    pass
