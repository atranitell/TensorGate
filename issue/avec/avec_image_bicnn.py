# -*- coding: utf-8 -*-
""" regression task for image and optical flow
    updated: 2018/04/10
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

slim = tf.contrib.slim


class AVEC_IMAGE_BICNN(context.Context):

  def __init__(self, config):
    context.Context.__init__(self, config)

  # def _net(self, data):
  #   """
  #   OPT -> feature1 -> logit1
  #          feature2 |
  #   RGB -> feature1 | -> logit
  #          logit1   |
  #   """
  #   data = tf.unstack(data, axis=1)
  #   _, net0 = network(data[0], self.config, self.phase, 'RBG', False)
  #   logit1, net1 = network(data[1], self.config, self.phase, 'OPT', False)
  #   net = tf.concat([tf.squeeze(net0['global_pool'], [1, 2]),
  #                    tf.squeeze(net1['global_pool'], [1, 2]), logit1], axis=1)
  #   logit = tf.contrib.layers.fully_connected(
  #       net, 1,
  #       biases_initializer=None,
  #       weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
  #       weights_regularizer=None,
  #       activation_fn=None,
  #       scope='logits')
  #   return logit, net

  def _net(self, data):
    """
    RGB -> feature1 -> logit1
           feature2 |
    OPT -> feature1 | -> logit
           logit1   |
    """
    data = tf.unstack(data, axis=1)
    logit0, net0 = network(data[0], self.config, self.phase, 'RBG', False)
    logit1, net1 = network(data[1], self.config, self.phase, 'OPT', False)
    net = tf.concat([tf.squeeze(net0['global_pool'], [1, 2]),
                     tf.squeeze(net1['global_pool'], [1, 2]), logit0], axis=1)
    logit = tf.contrib.layers.fully_connected(
        net, 1,
        biases_initializer=None,
        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
        weights_regularizer=None,
        activation_fn=None,
        scope='logits')
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

    variables.print_trainable_list()

    # hooks
    self.add_hook(self.snapshot.init())
    self.add_hook(self.summary.init())
    self.add_hook(context.Running_Hook(
        config=self.config.log,
        step=global_step,
        keys=['loss', 'mae', 'rmse'],
        values=[loss, mae, rmse],
        func_test=self.test,
        func_val=None))

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
    path = tf.unstack(path, axis=1)
    path = path[0]
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
    path = tf.unstack(path, axis=1)
    path = path[0]

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
