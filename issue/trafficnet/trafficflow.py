# -*- coding: utf-8 -*-
""" regression task for image
    updated: 2017/11/19
"""
import tensorflow as tf
from core.data.factory import loads
from core.network.factory import network
from core.network.rnn.cnn_brnn import cnn_brnn
from core.loss import l2
from core.solver import updater
from core.solver import context
from core.utils.variables import variables
from core.utils.filesystem import filesystem
from core.utils.string import string
from core.utils.logger import logger


class trafficflow(context.Context):

  def __init__(self, config):
    context.Context.__init__(self, config)

  def _net(self, data):
    # logit, net = cnn_brnn(data, self.config.net, self.is_train)
    logit, net = network(data, self.config, self.phase)
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
    data, labels, path = loads(self.config)
    # data = tf.reduce_sum(data, axis=3)
    # data = tf.expand_dims(data, axis=3)
    # data = tf.sign(data)

    label, cond = tf.unstack(labels, axis=1)
    # get network
    logit, net = self._net(data)
    cond = tf.reshape(cond, [self.data.batchsize, 1])
    label = tf.reshape(label, [self.data.batchsize, 1])
    label = label * cond
    # label = label * (cond - 234.) / (689. - 234.)
    # get loss
    loss, mae, rmse = self._loss(logit, label)

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
        keys=['loss', 'mae', 'rmse'],
        values=[loss, mae, rmse],
        func_test=self.test,
        func_val=None)

    # monitor session
    with tf.train.MonitoredTrainingSession(
            hooks=[running_hook, snapshot_hook, summary_hook,
                   tf.train.NanTensorHook(loss)],
            save_checkpoint_secs=None,
            save_summaries_steps=None) as sess:

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
    data, labels, path = loads(self.config)
    # data = tf.reduce_sum(data, axis=3)
    # data = tf.expand_dims(data, axis=3)
    # data = tf.sign(data)

    label, cond = tf.unstack(labels, axis=1)
    # get network
    logit, net = self._net(data)
    cond = tf.reshape(cond, [self.data.batchsize, 1])
    logit = tf.reshape(logit, [self.data.batchsize, 1])
    logit = logit / cond

    # total_num
    total_num = self.data.total_num
    batchsize = self.data.batchsize

    # get loss
    # logit = tf.reshape(logit, [self.data.batchsize, 1])
    # logit = logit * (689.0 - 234.0) / (cond - 234.0)
    loss, mae, rmse = self._loss(logit, label)

    # get saver
    saver = tf.train.Saver()
    with tf.Session() as sess:
      # get latest checkpoint
      global_step = self.snapshot.restore(sess, saver)

      # output to file
      info = string.concat(
          batchsize, [path, label, logit * self.data.range])
      with open(test_dir + '%s.txt' % global_step, 'wb') as fw:
        with context.QueueContext(sess):
          # Initial some variables
          num_iter = int(total_num / batchsize)
          mean_loss, mean_mae, mean_rmse = 0, 0, 0

          for _ in range(num_iter):
            # running session to acuqire value
            _loss, _mae, _rmse, _info = sess.run([loss, mae, rmse, info])
            mean_loss += _loss
            mean_mae += _mae
            mean_rmse += _rmse
            # save tensor info to text file
            [fw.write(_line + b'\r\n') for _line in _info]

          # statistic
          mean_loss = 1.0 * mean_loss / num_iter
          mean_mae = 1.0 * mean_mae / num_iter
          mean_rmse = 1.0 * mean_rmse / num_iter

      # display results on screen
      keys = ['total sample', 'num batch', 'loss', 'mae', 'rmse']
      vals = [total_num, num_iter, mean_loss, mean_mae, mean_rmse]
      logger.test(logger.iters(int(global_step), keys, vals))

      # write to summary
      self.summary.adds(global_step=global_step,
                        tags=['test/loss', 'test/mae', 'test/rmse'],
                        values=[mean_loss, mean_mae, mean_rmse])

      self._exit_()
      return mean_mae

  def val(self):
    pass