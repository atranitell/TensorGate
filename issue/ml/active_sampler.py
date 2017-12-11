# -*- coding: utf-8 -*-
""" classification task for image
    updated: 2017/11/19
"""
import tensorflow as tf
from core.database.factory import loads
from core.network.factory import network
from core.loss import softmax
from core.solver import updater
from core.solver import variables
from core import utils
from core.utils.logger import logger
from core.utils.profiler import Profiler
from issue import context

from core.database.database import DataBase
from core.database import data_entry
from PIL import Image
import numpy as np
import random


class active_sampler(context.Context):

  def __init__(self, config):
    context.Context.__init__(self, config)
    self.test_database = None

  def _net(self, data):
    data = tf.expand_dims(data, axis=3)
    logit, net = network(data, self.config, self.phase)
    return logit, net

  def _loss(self, logit, label):
    logit = tf.reshape(logit, [self.data.batchsize, self.data.num_classes])
    loss_batch = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=label, logits=logit, name='loss_batch')
    loss = tf.reduce_mean(loss_batch, name='loss')
    error, pred = softmax.get_error(logit, label)
    return loss, error, pred, loss_batch

  def train(self):
    """
    """
    # set phase
    self._enter_('train')

    database = DataBase(self.config)
    data, label, path = database.loads()

    # get network
    logit, net = self._net(data)
    # get loss
    loss, error, pred, loss_batch = self._loss(logit, label)
    loss_batch = tf.nn.softmax(loss_batch)

    # update
    global_step = tf.train.create_global_step()
    train_op = updater.default(self.config, loss, global_step)

    # for storage
    saver = tf.train.Saver(var_list=variables.all())

    # hooks
    snapshot_hook = self.snapshot.init()
    summary_hook = self.summary.init()
    running_hook = context.Running_Hook(
        config=self.config.log,
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

      # restore model: if checkpoint does not exited, do nothing
      self.snapshot.restore(sess, saver)

      # Profile
      # Profiler.time_memory(self.config['output_dir'], sess, train_op)
      while not sess.should_stop():
        feeds = database.next_batch(data, label, path)
        _, _loss = sess.run(
            [train_op, loss_batch],
            feed_dict=feeds)
        # print(_loss)
        database.gen_prob(_loss)

  def test(self):
    """
    """
    # save current context
    self._enter_('test')

    # create a folder to save
    test_dir = utils.filesystem.mkdir(self.config.output_dir + '/test/')

    # get data pipeline

    # backup database
    if self.test_database is None:
      database = DataBase(self.config)
      self.test_database = database
    else:
      self.test_database.reset_index()
    data, label, path = self.test_database.loads()

    # total_num
    total_num = self.data.total_num
    batchsize = self.data.batchsize
    # get network
    logit, net = self._net(data)
    # get loss
    loss, error, pred, loss_batch = self._loss(logit, label)

    # get saver
    saver = tf.train.Saver()
    with tf.Session() as sess:
      # get latest checkpoint
      global_step = self.snapshot.restore(sess, saver)

      # output to file
      info = utils.string.concat(batchsize, [path, label, pred])
      with open(test_dir + '%s.txt' % global_step, 'wb') as fw:
        with context.QueueContext(sess):
          # Initial some variables
          num_iter = int(total_num / batchsize)
          mean_err, mean_loss = 0, 0

          for _ in range(num_iter):
            # running session to acuqire value
            feeds = self.test_database.next_batch(data, label, path)
            _loss, _err, _info = sess.run([loss, error, info], feed_dict=feeds)
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
