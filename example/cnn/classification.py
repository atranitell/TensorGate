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
from core.utils.filesystem import filesystem
from core.utils.string import string
from core.utils.logger import logger
from core.utils.profiler import Profiler
from issue import context


class classification(context.Context):

  def __init__(self, config):
    context.Context.__init__(self, config)

  def _net(self, data):
    logit, net = network(data, self.config, self.phase)
    return logit, net

  def _loss(self, logit, label):
    loss, logit = softmax.get_loss(logit, label, self.config)
    error, pred = softmax.get_error(logit, label)
    return loss, error, pred

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
    loss, error, pred = self._loss(logit, label)

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
    loss, error, pred = self._loss(logit, label)

    # get saver
    saver = tf.train.Saver()
    with tf.Session() as sess:
      # get latest checkpoint
      global_step = self.snapshot.restore(sess, saver)

      # output to file
      info = string.concat(batchsize, [path, label, pred])
      with open(test_dir + '%s.txt' % global_step, 'wb') as fw:
        with context.QueueContext(sess):
          # Initial some variables
          num_iter = int(total_num / batchsize)
          mean_err, mean_loss = 0, 0

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
