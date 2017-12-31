# -*- coding: utf-8 -*-
""" Conditional GAN
    updated: 2017/11/22
"""
import tensorflow as tf
from core.database.factory import loads
from core.network.vaes import kin_vae
from core.solver import updater
from core.solver import variables
from core.loss import cosine
from core import utils
from core.utils.logger import logger
from issue import context
import numpy as np


class LFW(context.Context):
  """ 1E - 2G - 1D
  """

  def __init__(self, config):
    context.Context.__init__(self, config)

  def _encoder(self, x, reuse=None):
    return kin_vae.encoder(x, self.config.net.z_dim, self.is_train, reuse)

  def _generator(self, z, cond, reuse=None):
    return kin_vae.generator(z, cond, self.is_train, reuse, 128)

  def _discriminator(self, x, cond, reuse=None):
    return kin_vae.discriminator(x, cond, self.data.num_classes,
                                 self.is_train, reuse)

  def _loss_vae(self, real, fake, mu, sigma):
    """
    real: real images
    fake: generative images
    mu: the mean of encoding real images
    sigma: the std of encoding real images
    """
    marginal_likelihood = tf.reduce_sum(
        real * tf.log(fake) + (1 - real) * tf.log(1 - fake), [1, 2])
    KL_divergence = 0.5 * tf.reduce_sum(
        tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, [1])
    return marginal_likelihood, KL_divergence

  def _loss_vae_reduce(self, marginal_likelihood, KL_divergence):
    neg_loglikelihood = -tf.reduce_mean(marginal_likelihood)
    KL_divergence = tf.reduce_mean(KL_divergence)
    ELBO = -neg_loglikelihood - KL_divergence
    return -ELBO

  def _loss_gan(self, D_F, D_R):
    """
    D_F: discriminator logit for fake image
    D_R: discriminator logit for real image
    """
    D_loss_F = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(D_F), logits=D_F))
    D_loss_R = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(D_R), logits=D_R))
    D_loss = D_loss_F + D_loss_R
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(D_F), logits=D_F))
    return D_loss, G_loss

  def _loss_metric(self, feat_x, feat_y, label):
    return cosine.get_loss(feat_x, feat_y, label,
                           self.data.batchsize,
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
    mu2, sigma2, feat2 = self._encoder(real2, True)

    z1 = mu1 + sigma1 * tf.random_normal(tf.shape(mu1))
    z1 = self._generator(z1, cond)
    fake1 = tf.clip_by_value(z1, 1e-8, 1 - 1e-8)

    # discriminator
    D_fake1 = self._discriminator(fake1, cond)
    D_real2 = self._discriminator(real2, cond, reuse=True)

    # loss for encoder
    R_loss, _ = self._loss_metric(feat1, feat2, label)
    # loss for genertor
    E1_l1, E1_l2 = self._loss_vae(real1, fake1, mu1, sigma1)
    E2_l1, E2_l2 = self._loss_vae(real2, fake1, mu1, sigma1)
    label = tf.to_float(tf.reshape(label, [self.config.data.batchsize, 1]))
    E1_loss = (1 + label) * E1_l1 + (1 - label) * E2_l1
    E2_loss = (1 + label) * E1_l2 + (1 - label) * E2_l2
    E_loss = self._loss_vae_reduce(E1_loss, E2_loss)

    # loss for discriminator
    D_loss, G_loss = self._loss_gan(D_fake1, D_real2)
    # sum
    loss = E_loss + D_loss + G_loss + R_loss

    # # allocate two optimizer
    global_step = tf.train.create_global_step()

    var_e = variables.select_vars('encoder')
    var_g = variables.select_vars('generator')
    var_d = variables.select_vars('discriminator')

    op1 = updater.default(self.config, loss, global_step, var_e, 0)
    op2 = updater.default(self.config, loss, None, var_g, 1)
    op3 = updater.default(self.config, loss, None, var_d, 0)
    train_op = tf.group(op1, op2, op3)

    # update at the same time
    saver = tf.train.Saver(var_list=variables.all())

    # hooks
    snapshot_hook = self.snapshot.init()
    summary_hook = self.summary.init()
    running_hook = context.Running_Hook(
        config=self.config.log,
        step=global_step,
        keys=['E', 'D', 'G', 'R'],
        values=[E_loss, D_loss, G_loss, R_loss],
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
          # utils.image.save_batchs(
          #     image_list=[_cf, _c1, _p1, _p2],
          #     batchsize=batchsize, dstdir=dstdir, step=step,
          #     name_list=['_cf', '_c1', '_p1', '_p2'])
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
