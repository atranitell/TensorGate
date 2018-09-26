# Copyright 2017 The KaiJIN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""2018/2/25 AVEC2014 """

import tensorflow as tf
from gate import context
from gate.data.data_factory import load_data
from gate.net.net_factory import net_graph
from gate.solver import updater
from gate.layer import l2
from gate.utils import variable
from gate.utils import filesystem
from gate.utils import string
from gate.utils.logger import logger
from gate.utils.heatmap import HeatMap
from samples.avec2014.utils import get_accurate_from_file

from pycocotools import mask as maskUtils
import numpy as np
import cv2


class AVEC2014_68_IMG_CNN(context.Context):
  """ Use normal cnn training single image.
  """

  def __init__(self, config):
    context.Context.__init__(self, config)

  def segment(self, image, rle):
    loss = []
    for i in range(self.batchsize):
      rle_i = rle[:, i]
      image_i = image[i, :, :, :]
      rle_face = [
          rle_i[0], rle_i[1],  # p0
          rle_i[3], rle_i[4],  # p1
          rle_i[6], rle_i[7],  # p2
          rle_i[9], rle_i[10],  # p3
          rle_i[12], rle_i[13],  # p4
          rle_i[15], rle_i[16],  # p5
          rle_i[18], rle_i[19],  # p6
          rle_i[21], rle_i[22],  # p7
          rle_i[24], rle_i[25],  # p8
          rle_i[27], rle_i[28],  # p9
          rle_i[30], rle_i[31],  # p10
          rle_i[33], rle_i[34],  # p11
          rle_i[36], rle_i[37],  # p12
          rle_i[39], rle_i[40],  # p13
          rle_i[42], rle_i[43],  # p14
          rle_i[45], rle_i[46],  # p15
          rle_i[48], rle_i[49],  # p16
          # top
          rle_i[78], rle_i[79],  # p26
          rle_i[75], rle_i[76],  # p25
          rle_i[72], rle_i[73],  # p24
          rle_i[69], rle_i[70],  # p23
          rle_i[66], rle_i[67],  # p22
          rle_i[63], rle_i[64],  # p21
          rle_i[60], rle_i[61],  # p20
          rle_i[57], rle_i[58],  # p19
          rle_i[54], rle_i[55],  # p18
          rle_i[51], rle_i[52],  # p17
      ]
      mask = maskUtils.decode(maskUtils.frPyObjects(
          [rle_face], self.data.configs[0].raw_height,
          self.data.configs[0].raw_width))
      total_area = np.count_nonzero(mask)
      loss_i = float(np.sum(image_i * mask))/total_area
      loss.append(loss_i)
    return np.array(loss, dtype=np.float)

  def _net(self, data, label):
    net, end_points = net_graph(data, self.config.net[0], self.phase)
    net = tf.expand_dims(tf.reduce_mean(net, axis=3), axis=3)
    net = tf.image.resize_bicubic(net, [self.data.configs[0].raw_height,
                                        self.data.configs[0].raw_width])
    net = tf.py_func(self.segment, [net, label], tf.float32)
    tf.stop_gradient(net)
    # with tf.Session() as sess:
    #   sess.run(tf.global_variables_initializer())
    #   tf.train.start_queue_runners()
    #   print(sess.run(net))
    return net, end_points

  def _loss(self, logit, label):
    loss = l2.loss(logit, label, self.config)
    return loss

  def _error(self, logit, label):
    mae, rmse = l2.error(logit, label, self.config)
    return mae, rmse

  @context.graph_phase_wrapper()
  def train(self):
    # load data
    image, labels, _ = load_data(self.config)
    label = tf.unstack(labels, axis=1)
    # load net
    logit, _ = self._net(image, label)
    # compute loss
    loss = self._loss(logit, label[204])
    # compute error
    mae, rmse = self._error(logit, label[204])
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

  @context.graph_phase_wrapper()
  def test(self):
    # create a folder to save
    test_dir = filesystem.mkdir(self.config.output_dir + '/test/')
    # get data
    image, labels, path = load_data(self.config)
    label = tf.unstack(labels, axis=1)
    # get net
    logit, _ = self._net(image)
    # output to file
    info = string.concat(
        self.batchsize, [path, label[204], logit*self.data.span])

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
      return _rmse

  @context.graph_phase_wrapper()
  def heatmap(self):
    with tf.Graph().as_default():
      self.heatmap_gap()
    with tf.Graph().as_default():
      self.heatmap_cam()
    with tf.Graph().as_default():
      self.heatmap_gb()
    with tf.Graph().as_default():
      self.heatmap_guided_cam()

  def heatmap_gap(self):
    """ HEATMAP BY USING GAP
    """
    # save current context
    self._enter_('heatmap')
    # create a folder to save
    heatmap_dir = filesystem.mkdir(self.config.output_dir + '/heatmap_gap/')
    # get data pipeline
    data, label, path = load_data(self.config)

    # get network
    logit, net = self._net(data)
    loss = self._loss(logit, label)

    # heatmap
    heatmap = HeatMap(self.data.configs[0].raw_height,
                      self.data.configs[0].raw_width)
    x = net['gap_conv']
    w = variable.select_vars('resnet_v2_50/logits/weights')[0]

    # get saver
    saver = tf.train.Saver()
    with context.DefaultSession() as sess:
      self.snapshot.restore(sess, saver)
      with context.QueueContext(sess):
        for _ in range(int(self.data.total_num / self.data.batchsize)):
          _x, _w, _path = sess.run([x, w, path])
          dstlist, srclist = heatmap.make_paths(heatmap_dir, _path)
          heatmap.gap(_x, _w[0], [0], dstlist, srclist, True)
    logger.info('HeatMAP Image has saved in %s' % heatmap_dir)
    self._exit_()
    return 0

  def heatmap_gb(self):
    """ HEATMAP BY USING guided backpropagation
    """
    # save current context
    self._enter_('heatmap')
    # create a folder to save
    heatmap_dir = filesystem.mkdir(self.config.output_dir + '/heatmap_gb/')
    # get data pipeline
    data, label, path = load_data(self.config)

    # get network
    logit, net = self._net(data)
    loss = self._loss(logit, label)

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
          dstlist, srclist = heatmap.make_paths(heatmap_dir, _path)
          heatmap.guided_backpropagation(_gb, dstlist)
    logger.info('HeatMAP Image has saved in %s' % heatmap_dir)
    self._exit_()
    return 0

  def heatmap_cam(self):
    """ HEATMAP BY USING CAM
    """
    # save current context
    self._enter_('heatmap')
    # create a folder to save
    heatmap_dir = filesystem.mkdir(self.config.output_dir + '/heatmap_cam/')
    # get data pipeline
    data, label, path = load_data(self.config)

    # get network
    logit, net = self._net(data)
    loss = self._loss(logit, label)

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
          dstlist, srclist = heatmap.make_paths(heatmap_dir, _path)
          heatmap.grad_cam(_x, _g, dstlist, srclist, True)
    logger.info('HeatMAP Image has saved in %s' % heatmap_dir)
    self._exit_()
    return 0

  def heatmap_guided_cam(self):
    """ HEATMAP BY USING GUIDED CAM
    """
    # save current context
    self._enter_('heatmap')
    # create a folder to save
    heatmap_dir = filesystem.mkdir(self.config.output_dir + '/heatmap_gb_cam/')
    # get data pipeline
    data, label, path = load_data(self.config)

    # get network
    logit, net = self._net(data)
    loss = self._loss(logit, label)

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
          dstlist, srclist = heatmap.make_paths(heatmap_dir, _path)
          heatmap.guided_grad_cam(_x, _g, _gb, dstlist, srclist, True)
    logger.info('HeatMAP Image has saved in %s' % heatmap_dir)
    self._exit_()
    return 0
