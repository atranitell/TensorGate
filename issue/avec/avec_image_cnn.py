# -*- coding: utf-8 -*-
""" regression task for image
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
from core.utils.heatmap import HeatMap

from issue.avec.avec_utils import get_accurate_from_file


class AVEC_IMAGE_CNN(context.Context):

  def __init__(self, config):
    context.Context.__init__(self, config)

  def _net(self, data):
    # for old version, the name should keep 'net'
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

  def heatmap(self):
    self.heatmap_guided_cam()

  def heatmap_gap(self):
    """ HEATMAP BY USING GAP
    """
    # save current context
    self._enter_('test')
    # create a folder to save
    test_dir = filesystem.mkdir(self.config.output_dir + '/heatmap_gap/')
    # get data pipeline
    data, label, path = loads(self.config)

    # get network
    logit, net = self._net(data)
    loss, mae, rmse = self._loss(logit, label)

    # heatmap
    heatmap = HeatMap(self.data.configs[0].raw_height,
                      self.data.configs[0].raw_width)
    x = net['gap_conv']
    w = variables.select_vars('net/resnet_v2_50/logits/weights')[0]

    # get saver
    saver = tf.train.Saver()
    with context.DefaultSession() as sess:
      self.snapshot.restore(sess, saver)
      with context.QueueContext(sess):
        for _ in range(int(self.data.total_num / self.data.batchsize)):
          _x, _w, _path = sess.run([x, w, path])
          dstlist, srclist = heatmap.make_paths(test_dir, _path)
          heatmap.gap(_x, _w[0], [0], dstlist, srclist, True)

    self._exit_()
    return 0

  def heatmap_gb(self):
    """ HEATMAP BY USING guided backpropagation
    """
    # save current context
    self._enter_('test')
    # create a folder to save
    test_dir = filesystem.mkdir(self.config.output_dir + '/heatmap_gb/')
    # get data pipeline
    data, label, path = loads(self.config)

    # get network
    logit, net = self._net(data)
    loss, mae, rmse = self._loss(logit, label)

    # heatmap
    heatmap = HeatMap(self.data.configs[0].raw_height,
                      self.data.configs[0].raw_width)
    gb_grad = tf.gradients(logit, data)[0]

    # get saver
    saver = tf.train.Saver()
    with context.DefaultSession() as sess:
      self.snapshot.restore(sess, saver)
      with context.QueueContext(sess):
        for _ in range(int(self.data.total_num / self.data.batchsize)):
          _gb, _path = sess.run([gb_grad, path])
          dstlist, srclist = heatmap.make_paths(test_dir, _path)
          heatmap.guided_backpropagation(_gb, dstlist)

    self._exit_()
    return 0

  def heatmap_cam(self):
    """ HEATMAP BY USING CAM
    """
    # save current context
    self._enter_('test')
    # create a folder to save
    test_dir = filesystem.mkdir(self.config.output_dir + '/heatmap_cam/')
    # get data pipeline
    data, label, path = loads(self.config)

    # get network
    logit, net = self._net(data)
    loss, mae, rmse = self._loss(logit, label)

    # heatmap
    heatmap = HeatMap(self.data.configs[0].raw_height,
                      self.data.configs[0].raw_width)
    x = net['gap_conv']
    g = tf.gradients(logit, x)[0]

    # get saver
    saver = tf.train.Saver()
    with context.DefaultSession() as sess:
      self.snapshot.restore(sess, saver)
      with context.QueueContext(sess):
        for _ in range(int(self.data.total_num / self.data.batchsize)):
          _x, _g, _path = sess.run([x, g, path])
          dstlist, srclist = heatmap.make_paths(test_dir, _path)
          heatmap.grad_cam(_x, _g, dstlist, srclist, True)

    self._exit_()
    return 0

  def heatmap_guided_cam(self):
    """ HEATMAP BY USING GUIDED CAM
    """
    # save current context
    self._enter_('test')
    # create a folder to save
    test_dir = filesystem.mkdir(self.config.output_dir + '/heatmap_gb_cam/')
    # get data pipeline
    data, label, path = loads(self.config)

    # get network
    logit, net = self._net(data)
    loss, mae, rmse = self._loss(logit, label)

    # heatmap
    heatmap = HeatMap(self.data.configs[0].output_height,
                      self.data.configs[0].output_width)
    x = net['gap_conv']
    g = tf.gradients(logit, x)[0]
    gb_grad = tf.gradients(logit, data)[0]

    # get saver
    saver = tf.train.Saver()
    with context.DefaultSession() as sess:
      self.snapshot.restore(sess, saver)
      with context.QueueContext(sess):
        for _ in range(int(self.data.total_num / self.data.batchsize)):
          _x, _g, _gb, _path = sess.run([x, g, gb_grad, path])
          dstlist, srclist = heatmap.make_paths(test_dir, _path)
          heatmap.guided_grad_cam(_x, _g, _gb, dstlist, srclist, True)

    self._exit_()
    return 0
