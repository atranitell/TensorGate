# -*- coding: utf-8 -*-
""" updated: 2017/11/22
"""
import tensorflow as tf
from core.database.factory import loads
from core.solver import updater
from core.solver import variables
from issue import context
from issue.kinface.kinvae_bidirect import KINVAE_BIDIRECT



class KINVAE_BIDIRECT3(KINVAE_BIDIRECT):
  """ 2E - 2G - 2D
  """

  def __init__(self, config):
    KINVAE_BIDIRECT.__init__(self, config)

  def _network(self, x, y):
    with tf.variable_scope('net1'):
      x_mu, x_sigma, x_feat = self._encoder(x)
    with tf.variable_scope('net2'):
      y_mu, y_sigma, y_feat = self._encoder(y)
    return x_feat, y_feat

  def train(self):
    """ """
    # set phase
    self._enter_('train')

    # get data pipeline
    data, info, _ = loads(self.config)
    c1_real, p1_real, c2_real, p2_real = tf.unstack(data, axis=1)
    label, cond = tf.unstack(info, axis=1)

    # encode image to a vector
    with tf.variable_scope('net1'):
      c1_mu, c1_sigma, feat_c1 = self._encoder(c1_real)
    with tf.variable_scope('net2'):
      p2_mu, p2_sigma, feat_p2 = self._encoder(p2_real)

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
      D_p1_real = self._discriminator(p1_real, cond, reuse=True)

    with tf.variable_scope('net2'):
      D_p2_fake = self._discriminator(p2_fake, cond)
      D_c2_real = self._discriminator(c2_real, cond, reuse=True)

    # loss for encoder
    R1_loss, _ = self._loss_metric(feat_c1, feat_p2, label)
    R2_loss, _ = self._loss_metric(D_c2_real, D_p1_real, label)
    R3_loss, _ = self._loss_metric(D_c1_fake, D_p2_fake, label)
    R_loss = R1_loss + R2_loss + R3_loss

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
