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


class KINVAE_BIDIRECT(context.Context):
  """ 1E - 2G - 1D
  """

  def __init__(self, config):
    context.Context.__init__(self, config)

  def _encoder(self, x, reuse=None):
    return kin_vae.encoder(x, self.config.net.z_dim, self.is_train, reuse)

  def _generator(self, z, cond, reuse=None):
    return kin_vae.generator(z, cond, self.is_train, reuse)

  def _discriminator(self, x, cond, reuse=None):
    return kin_vae.discriminator(x, cond, self.data.num_classes,
                                 self.is_train, reuse)

  def _network(self, c1_real, p2_real):
    c1_mu, c1_sigma, feat_c1 = self._encoder(c1_real)
    p2_mu, p2_sigma, feat_p2 = self._encoder(p2_real, True)
    return feat_c1, feat_p2

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
    c1_real, p1_real, c2_real, p2_real = tf.unstack(data, axis=1)
    label, cond = tf.unstack(info, axis=1)

    # encode image to a vector
    c1_mu, c1_sigma, feat_c1 = self._encoder(c1_real)
    p2_mu, p2_sigma, feat_p2 = self._encoder(p2_real, True)

    # children to parent
    with tf.variable_scope('net1'):
      c1_z = c1_mu + c1_sigma * tf.random_normal(tf.shape(c1_mu))
      c1_z = self._generator(c1_z, cond)
      c1_fake = tf.clip_by_value(c1_z, 1e-8, 1 - 1e-8)

    # parent to children
    with tf.variable_scope('net2'):
      p2_z = p2_mu + p2_sigma * tf.random_normal(tf.shape(p2_mu))
      p2_z = self._generator(p2_z, cond)
      p2_fake = tf.clip_by_value(p2_z, 1e-8, 1 - 1e-8)

    # discriminator
    D_c1_fake = self._discriminator(c1_fake, cond, reuse=False)
    D_p1_real = self._discriminator(p1_real, cond, reuse=True)

    D_p2_fake = self._discriminator(p2_fake, cond, reuse=True)
    D_c2_real = self._discriminator(c2_real, cond, reuse=True)

    # loss for encoder
    R_loss, _ = self._loss_metric(feat_c1, feat_p2, label)

    # loss for genertor
    E1_loss = self._loss_vae(p1_real, c1_fake, c1_mu, c1_sigma)
    E2_loss = self._loss_vae(c2_real, p2_fake, p2_mu, p2_sigma)
    E_loss = E1_loss + E2_loss

    # loss for discriminator
    D1_loss, G1_loss = self._loss_gan(D_c1_fake, D_p1_real)
    D2_loss, G2_loss = self._loss_gan(D_p2_fake, D_c2_real)
    D_loss = D1_loss + D2_loss
    G_loss = G1_loss + G2_loss

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
    self.add_hook(self.snapshot.init())
    self.add_hook(self.summary.init())
    self.add_hook(context.Running_Hook(
        config=self.config.log,
        step=global_step,
        keys=['E', 'D', 'G', 'R'],
        values=[E_loss, D_loss, G_loss, R_loss],
        func_test=self.test,
        func_val=None))

    with context.DefaultSession(self.hooks) as sess:
      self.snapshot.restore(sess, saver)
      while not sess.should_stop():
        sess.run(train_op)

  def _val_or_test(self, dstdir):
    """ COMMON FOR TRAIN AND VAL """
    # considering output train image
    data, info, path = loads(self.config)
    c1_real, p1_real, c2_real, p2_real = tf.unstack(data, axis=1)
    c1_path, p1_path, c2_path, p2_path = tf.unstack(path, axis=1)
    label, cond = tf.unstack(info, axis=1)

    # encode image to a vector
    feat_c1, feat_p2 = self._network(c1_real, p2_real)
    R_loss, loss = self._loss_metric(feat_c1, feat_p2, None)

    saver = tf.train.Saver()
    c1_zs, p2_zs = 0, 0

    with context.DefaultSession() as sess:
      step = self.snapshot.restore(sess, saver)

      info = utils.string.concat(
          self.batchsize,
          [c1_path, p1_path, c2_path, p2_path, label, loss])

      output = [info, feat_c1, feat_p2, label]
      fw = open(dstdir + '%s.txt' % step, 'wb')

      with context.QueueContext(sess):
        for i in range(self.epoch_iter):
          _info, _x, _y, _label = sess.run(output)
          self._write_feat_to_npy(i, _x, _y, _label)
          [fw.write(_line + b'\r\n') for _line in _info]
          c1_zs = _x if type(c1_zs) == int else np.row_stack((c1_zs, _x))
          p2_zs = _y if type(p2_zs) == int else np.row_stack((p2_zs, _y))

      fw.close()
      np.save(dstdir + '/' + step + '_c1.npy', c1_zs)
      np.save(dstdir + '/' + step + '_p2.npy', p2_zs)

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
          self.test_x, self.test_y, self.test_l, False)
      keys = ['val_error', 'thred', 'test_error']
      vals = [val_err, val_thed, test_err]
      logger.test(logger.iters(int(step) - 1, keys, vals))
      self._exit_()
