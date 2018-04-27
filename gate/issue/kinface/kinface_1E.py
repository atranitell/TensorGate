# -*- coding: utf-8 -*-
"""
DECTRION FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2018/2/25

--------------------------------------------------------

FOR KINFACE

"""

import tensorflow as tf
from gate import context
from gate.data.factory import get_data
from gate.solver import updater
from gate.util import variable
from gate.util import filesystem
from gate.util import string
from gate.util.logger import logger
from gate.issue.avec2014.utils import get_accurate_from_file

from gate.net.vae import kinvae
from gate.issue.kinface import kinbase
from gate.layer import similarity


class KINFACE_1E(kinbase.KINBASE):
  """ Use normal cnn training single image.
  """

  def __init__(self, config):
    kinbase.KINBASE.__init__(self, config)

  def _net(self, c1_real, p2_real):
    c1_mu, c1_sigma, feat_c1 = self._encoder(c1_real)
    p2_mu, p2_sigma, feat_p2 = self._encoder(p2_real, True)
    return feat_c1, feat_p2

  def train(self):
    self._enter_('train')
    with tf.Graph().as_default() as graph:
      # load data
      data, info, path = get_data(self.config)
      c1_real, p1_real, c2_real, p2_real = tf.unstack(data, axis=1)
      label, cond = tf.unstack(info, axis=1)
      # load net
      feat_c1, feat_p2 = self._net(c1_real, p2_real)
      loss, loss_batch = self._loss_metric(feat_c1, feat_p2, label)
      # update gradients
      global_step = tf.train.create_global_step()
      train_op = updater.default(self.config, loss, global_step)
      # add hooks
      self.add_hook(self.snapshot.init())
      self.add_hook(self.summary.init())
      self.add_hook(context.Running_Hook(
          config=self.config.log,
          step=global_step,
          keys=['R'],
          values=[loss],
          func_test=self.test,
          func_val=None))

      saver = tf.train.Saver(var_list=variable.all())
      with context.DefaultSession(self.hooks) as sess:
        self.snapshot.restore(sess, saver)
        while not sess.should_stop():
          sess.run(train_op)

    self._exit_()

  def _inference(self, output_dir):
    """ COMMON FOR TRAIN AND VAL """
    # considering output train image
    data, info, path = get_data(self.config)
    c1_real, p1_real, c2_real, p2_real = tf.unstack(data, axis=1)
    c1_path, p1_path, c2_path, p2_path = tf.unstack(path, axis=1)
    label, cond = tf.unstack(info, axis=1)

    # encode image to a vector
    feat_c1, feat_p2 = self._network(c1_real, p2_real)
    R_loss, loss = self._loss_metric(feat_c1, feat_p2, None)

    saver = tf.train.Saver()
    c1_zs, p2_zs = 0.0, 0.0

    with context.DefaultSession() as sess:
      step = self.snapshot.restore(sess, saver)
      info = string.concat(
          self.batchsize,
          [c1_path, p1_path, c2_path, p2_path, label, loss])

      output = [info, feat_c1, feat_p2, label]
      fw = open(output_dir + '%s.txt' % step, 'wb')

      with context.QueueContext(sess):
        for i in range(self.num_batch):
          _info, _x, _y, _label = sess.run(output)
          self._write_feat_to_npy(i, _x, _y, _label)
          [fw.write(_line + b'\r\n') for _line in _info]
          c1_zs = _x if isinstance(c1_zs, int) else np.row_stack((c1_zs, _x))
          p2_zs = _y if isinstance(p2_zs, int) else np.row_stack((p2_zs, _y))

      fw.close()
      np.save(output_dir + '/' + step + '_c1.npy', c1_zs)
      np.save(output_dir + '/' + step + '_p2.npy', p2_zs)

      return step

  def test(self):
    """ we need acquire threshold from validation first """
    with tf.Graph().as_default():
      self._enter_('val')
      _val_dir = filesystem.mkdir(self.config.output_dir + '/val/')
      _step = self._inference(_val_dir)
      self._exit_()

    with tf.Graph().as_default():
      self._enter_('test')
      _tst_dir = filesystem.mkdir(self.config.output_dir + '/test/')
      _step = self._inference(_tst_dir)
      val_err, val_thed, test_err = similarity.get_all_result(
          self.val_x, self.val_y, self.val_l,
          self.test_x, self.test_y, self.test_l, False)
      keys = ['val_error', 'thred', 'test_error']
      vals = [val_err, val_thed, test_err]
      logger.test(logger.iters(int(_step) - 1, keys, vals))
      self._exit_()

    # self._enter_('test')
    # # create a folder to save
    # test_dir = filesystem.mkdir(self.config.output_dir + '/test/')
    # # get data
    # image, label, path = get_data(self.config)
    # # get net
    # logit, _ = self._net(image)
    # # output to file
    # info = string.concat(self.batchsize, [path, label, logit*self.data.span])
    # saver = tf.train.Saver()

    # # running test
    # with context.DefaultSession() as sess:
    #   global_step = self.snapshot.restore(sess, saver)
    #   result_path = test_dir + '%s.txt' % global_step
    #   with open(result_path, 'wb') as fw:
    #     with context.QueueContext(sess):
    #       for _ in range(self.num_batch):
    #         _info = sess.run(info)
    #         [fw.write(_line + b'\r\n') for _line in _info]

    #   # display results on screen
    #   _mae, _rmse = get_accurate_from_file(result_path)
    #   keys = ['total sample', 'num batch', 'video_mae', 'video_rmse']
    #   vals = [self.total_num, self.num_batch, _mae, _rmse]
    #   logger.test(logger.iters(int(global_step), keys, vals))

    #   # write to summary
    #   self.summary.adds(global_step=global_step,
    #                     tags=['test/video_mae', 'test/video_rmse'],
    #                     values=[_mae, _rmse])

    #   self._exit_()
    #   return _rmse
