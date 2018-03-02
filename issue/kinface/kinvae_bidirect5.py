# -*- coding: utf-8 -*-
""" updated: 2017/11/22
"""
import tensorflow as tf
from core.data.factory import loads
from core.solver import updater
from core.solver import context
from core.utils.variables import variables
from issue.kinface.kinvae_bidirect2 import KINVAE_BIDIRECT2


class KINVAE_BIDIRECT5(KINVAE_BIDIRECT2):
  """ 1E - 2G - 2D
    C -> G(C) <---> C
    P -> G(P) <---> P
    without cosine loss
  """

  def __init__(self, config):
    KINVAE_BIDIRECT2.__init__(self, config)
    self.is_save_all_images = False
    self.is_save_batch_images = True
    self.is_save_feats = True

  def train(self):
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
    with tf.variable_scope('net1'):
      D_c1_fake = self._discriminator(c1_fake, cond)
      D_c1_real = self._discriminator(c1_real, cond, reuse=True)

    with tf.variable_scope('net2'):
      D_p2_fake = self._discriminator(p2_fake, cond)
      D_p2_real = self._discriminator(p2_real, cond, reuse=True)

    # loss for genertor
    E1_loss = self._loss_vae(c1_real, c1_fake, c1_mu, c1_sigma)
    E2_loss = self._loss_vae(p2_real, p2_fake, p2_mu, p2_sigma)
    E_loss = E1_loss + E2_loss

    # loss for discriminator
    D1_loss, G1_loss = self._loss_gan(D_c1_fake, D_c1_real)
    D2_loss, G2_loss = self._loss_gan(D_p2_fake, D_p2_real)
    D_loss = D1_loss + D2_loss
    G_loss = G1_loss + G2_loss

    loss = E_loss + D_loss + G_loss

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
        keys=['E', 'D', 'G'],
        values=[E_loss, D_loss, G_loss],
        func_test=self.test,
        func_val=None))

    with context.DefaultSession(self.hooks) as sess:
      self.snapshot.restore(sess, saver)
      while not sess.should_stop():
        sess.run(train_op)
