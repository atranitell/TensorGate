# -*- coding: utf-8 -*-
""" Guided Learning
    updated: 2018/03/11
"""
import tensorflow as tf
from core.data.factory import loads
from core.network.factory import network
from core.loss import softmax
from core.solver import updater
from core.solver import context
from core.utils.variables import variables
from core.utils.filesystem import filesystem
from core.utils.string import string
from core.utils.logger import logger
from core.utils.heatmap import HeatMap
from tensorflow.contrib import layers


class GUIDED_LEARNING(context.Context):

  def __init__(self, config):
    context.Context.__init__(self, config)

  def bn(self, x, is_training):
    return layers.batch_norm(
        inputs=x,
        decay=0.9,
        updates_collections=None,
        epsilon=1e-5,
        scale=True,
        is_training=is_training)

  def conv2d(self, x, filters, ksize, stride, name="conv2d"):
    return tf.layers.conv2d(
        inputs=x,
        filters=filters,
        kernel_size=ksize,
        strides=stride,
        padding='SAME',
        use_bias=False,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        kernel_regularizer=layers.l2_regularizer(0.0001),
        name=name)

  def fc(self, x, filters, name):
    return layers.fully_connected(
        x, filters,
        biases_initializer=None,
        weights_initializer=layers.xavier_initializer(),
        weights_regularizer=None,
        activation_fn=None,
        scope=name)

  def guided_network(self, x, num_classes, is_training, reuse=None):
    """
    """
    with tf.variable_scope('guided_network', reuse=reuse):
      end_points = {}
      # 224x224x3
      net = self.conv2d(x, 64, (7, 7), (2, 2), name='conv1')
      end_points['conv1'] = net

      # 112x112x64
      net = self.conv2d(net, 128, (7, 7), (2, 2), name='conv2')
      end_points['conv2'] = net

      # 56x56x128
      net = self.conv2d(net, 256, (7, 7), (2, 2), name='conv3')
      end_points['conv3'] = net

      # 28x28x256
      net = self.conv2d(net, 512, (7, 7), (2, 2), name='conv4')
      end_points['conv4'] = net

      # 14x14x512
      net = self.conv2d(net, 512, (7, 7), (2, 2), name='conv5')
      end_points['conv5'] = net

      # 7x7x512
      # net = self.conv2d(net, 4096, (1, 1), (1, 1), name='conv6')
      # end_points['conv6'] = net

      # 7x7x4096
      net = tf.reduce_mean(net, [1, 2])

      # 1x1x4096
      logits = self.fc(net, num_classes, 'fc')
      end_points['fc'] = logits

      return logits, end_points

  def _net(self, data):
    logit_s, net_s = self.guided_network(data, 1000, self.is_train)
    logit_t, net_t = network(data, self.config, 'test')
    return logit_s, net_s, logit_t, net_t

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
    logit_s, net_s, logit_t, net_t = self._net(data)

    l1 = tf.reduce_mean(net_s['conv1'] - net_t['vgg_16/pool1'], [0, 1, 2, 3])
    l2 = tf.reduce_mean(net_s['conv2'] - net_t['vgg_16/pool2'], [0, 1, 2, 3])
    l3 = tf.reduce_mean(net_s['conv3'] - net_t['vgg_16/pool3'], [0, 1, 2, 3])
    l4 = tf.reduce_mean(net_s['conv4'] - net_t['vgg_16/pool4'], [0, 1, 2, 3])
    l5 = tf.reduce_mean(net_s['conv5'] - net_t['vgg_16/pool5'], [0, 1, 2, 3])
    l6 = tf.reduce_mean(logit_s - logit_t, [0, 1])
    loss = l1 + l2 + l3 + l4 + l5 + l6

    # get loss
    _, error, pred = self._loss(logit_s, label)

    # update
    global_step = tf.train.create_global_step()
    trained_vars = variables.select_vars('guided_network')
    train_op = updater.default(self.config, loss, global_step, trained_vars)

    # for storage
    saver = tf.train.Saver(var_list=variables.all())

    # hooks
    self.add_hook(self.snapshot.init())
    self.add_hook(self.summary.init())
    self.add_hook(context.Running_Hook(
        config=self.config.log,
        step=global_step,
        keys=['loss', 'error'],
        values=[loss, error],
        func_test=self.test,
        func_val=None))

    # monitor session
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

    # network
    logit_s, net_s, logit_t, net_t = self._net(data)
    loss, error, pred = self._loss(logit_s, label)

    # prepare
    info = string.concat(batchsize, [path, label, pred])
    num_iter = int(total_num / batchsize)
    mean_err, mean_loss = 0, 0

    # get saver
    saver = tf.train.Saver()
    with context.DefaultSession() as sess:
      global_step = self.snapshot.restore(sess, saver)
      with open(test_dir + '%s.txt' % global_step, 'wb') as fw:
        with context.QueueContext(sess):
          for _ in range(num_iter):
            _loss, _err, _info = sess.run([loss, error, info])
            mean_loss += _loss
            mean_err += _err
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
