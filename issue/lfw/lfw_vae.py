# -*- coding: utf-8 -*-
""" updated: 2018/01/09
"""
import tensorflow as tf
from core.database.factory import loads
from core.network.vaes import lfw_vae
from core.solver import updater
from core.solver import variables
from core.loss import cosine
from core import utils
from core.utils.logger import logger
from issue import context
import numpy as np


class LFW_VAE(context.Context):
  """ 1E - 1G
  """

  def __init__(self, config):
    context.Context.__init__(self, config)

  def _encoder(self, x, reuse=None):
    return lfw_vae.encoder(x, self.config.net.z_dim, self.is_train, reuse)

  def _generator(self, z, cond, reuse=None):
    return lfw_vae.generator(z, cond, self.is_train, reuse)

  def _loss_vae(self, mu, sigma):
    """
    real: real images
    fake: generative images
    mu: the mean of encoding real images
    sigma: the std of encoding real images
    """
    KL_divergence = 0.5 * tf.reduce_sum(
        tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, [1])
    KL_divergence = tf.reduce_mean(KL_divergence)
    ELBO = - KL_divergence
    return -ELBO

  def _loss_metric(self, img_x, img_y, label):
    feat_x = tf.layers.flatten(img_x)
    feat_y = tf.layers.flatten(img_y)
    return cosine.get_loss(feat_x, feat_y, label,
                           self.batchsize,
                           is_training=self.is_train)

  def _write_feat_to_npy(self, idx, x, y, label):
    """ fast to record x1, x2, label to npy array """
    if self.phase == 'val':
      self.val_x = x if idx == 0 else np.row_stack((self.val_x, x))
      self.val_y = y if idx == 0 else np.row_stack((self.val_y, y))
      self.val_l = label if idx == 0 else np.append(self.val_l, label)
    elif self.phase == 'test':
      self.test_x = x if idx == 0 else np.row_stack((self.test_x, x))
      self.test_y = y if idx == 0 else np.row_stack((self.test_y, y))
      self.test_l = label if idx == 0 else np.append(self.test_l, label)

  def train(self):
    """ """
    # set phase
    self._enter_('train')

    # get data pipeline
    data, info, _ = loads(self.config)
    real1, real2 = tf.unstack(data, axis=1)
    label, cond = tf.unstack(info, axis=1)

    mu1, sigma1, feat1 = self._encoder(real1)

    z1 = mu1 + sigma1 * tf.random_normal(tf.shape(mu1))
    z1 = self._generator(z1, cond)
    fake1 = tf.clip_by_value(z1, 1e-8, 1 - 1e-8)

    # loss for encoder
    G_loss, _ = self._loss_metric(fake1, real2, label)
    # loss for genertor
    E_loss = self._loss_vae(mu1, sigma1)
    
    # sum
    loss = E_loss + G_loss

    # # allocate two optimizer
    global_step = tf.train.create_global_step()

    var_e = variables.select_vars('encoder')
    var_g = variables.select_vars('generator')

    op1 = updater.default(self.config, loss, global_step, var_e, 0)
    op2 = updater.default(self.config, loss, None, var_g, 1)
    train_op = tf.group(op1, op2)

    # update at the same time
    saver = tf.train.Saver(var_list=variables.all())

    # hooks
    snapshot_hook = self.snapshot.init()
    summary_hook = self.summary.init()
    running_hook = context.Running_Hook(
        config=self.config.log,
        step=global_step,
        keys=['E', 'G'],
        values=[E_loss, G_loss],
        func_test=self.test,
        func_val=None)

    # monitor session
    with tf.train.MonitoredTrainingSession(
            hooks=[running_hook, snapshot_hook, summary_hook],
            save_checkpoint_secs=None,
            save_summaries_steps=None) as sess:

      # restore model
      self.snapshot.restore(sess, saver)

      # running
      while not sess.should_stop():
        sess.run(train_op)

  def _val_or_test(self, dstdir):
    """ COMMON FOR TRAIN AND VAL """
    # considering output train image
    data, info, path = loads(self.config)
    real1, real2 = tf.unstack(data, axis=1)
    label, cond = tf.unstack(info, axis=1)
    path1, path2 = tf.unstack(path, axis=1)

    mu1, sigma1, feat1 = self._encoder(real1)
    mu2, sigma2, feat2 = self._encoder(real2, True)

    # encode image to a vector
    R_loss, loss = self._loss_metric(feat1, feat2, None)

    saver = tf.train.Saver()
    with tf.Session() as sess:
      step = self.snapshot.restore(sess, saver)
      info = utils.string.concat(
          self.data.batchsize, [path1, path2, label, loss])
      output = [info, feat1, feat2, label]
      fw = open(dstdir + '%s.txt' % step, 'wb')
      with context.QueueContext(sess):
        for i in range(int(self.data.total_num / self.data.batchsize)):
          _info, _x, _y, _label = sess.run(output)
          self._write_feat_to_npy(i, _x, _y, _label)
          [fw.write(_line + b'\r\n') for _line in _info]
      fw.close()
      return step

  def test(self):
    """ we need acquire threshold from validation first """
    with tf.Graph().as_default():
      self._enter_('val')
      val_dir = utils.filesystem.mkdir(self.config.output_dir + '/val/')
      self._val_or_test(val_dir)
      self._exit_()

    with tf.Graph().as_default():
      self._enter_('test')
      test_dir = utils.filesystem.mkdir(self.config.output_dir + '/test/')
      step = self._val_or_test(test_dir)
      val_err, val_thed, test_err = utils.similarity.get_all_result(
          self.val_x, self.val_y, self.val_l,
          self.test_x, self.test_y, self.test_l, True)
      keys = ['val_error', 'thred', 'test_error']
      vals = [val_err, val_thed, test_err]
      logger.test(logger.iters(int(step) - 1, keys, vals))
      self._exit_()
