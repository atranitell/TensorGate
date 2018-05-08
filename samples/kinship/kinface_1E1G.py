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
from gate.issue.kinface import kinbase


class KINFACE_1E1G(kinbase.KINBASE):

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
      c1_mu, c1_sigma, feat_c1 = self._encoder(c1_real)
      p2_mu, p2_sigma, feat_p2 = self._encoder(p2_real, True)

      # children to parent
      c1_z = c1_mu + c1_sigma * tf.random_normal(tf.shape(c1_mu))
      c1_z = self._generator(c1_z, cond)
      c1_fake = tf.clip_by_value(c1_z, 1e-8, 1 - 1e-8)

      # parent to children
      p2_z = p2_mu + p2_sigma * tf.random_normal(tf.shape(p2_mu))
      p2_z = self._generator(p2_z, cond, True)
      p2_fake = tf.clip_by_value(p2_z, 1e-8, 1 - 1e-8)

      # loss for genertor
      E1_loss = self._loss_vae(p1_real, c1_fake, c1_mu, c1_sigma)
      E2_loss = self._loss_vae(c2_real, p2_fake, p2_mu, p2_sigma)
      E_loss = E1_loss + E2_loss

      # summary loss
      loss = E_loss

      # update gradients
      global_step = tf.train.create_global_step()

      var_e = variable.select_vars('encoder')
      var_g = variable.select_vars('generator')

      op1 = updater.default(self.config, loss, global_step, var_e, 0)
      op2 = updater.default(self.config, loss, None, var_g, 1)
      train_op = tf.group(op1, op2)

      # add hooks
      self.add_hook(self.snapshot.init())
      self.add_hook(self.summary.init())
      self.add_hook(context.Running_Hook(
          config=self.config.log,
          step=global_step,
          keys=['E'],
          values=[E_loss],
          func_test=self.test,
          func_val=None))

      saver = tf.train.Saver(var_list=variable.all())
      with context.DefaultSession(self.hooks) as sess:
        self.snapshot.restore(sess, saver)
        while not sess.should_stop():
          sess.run(train_op)

    self._exit_()
