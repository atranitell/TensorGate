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


class KINFACE_2E(kinbase.KINBASE):

  def __init__(self, config):
    kinbase.KINBASE.__init__(self, config)

  def _net(self, c1_real, p2_real):
    with tf.variable_scope('net1'):
      c1_mu, c1_sigma, feat_c1 = self._encoder(c1_real)
    with tf.variable_scope('net2'):
      p2_mu, p2_sigma, feat_p2 = self._encoder(p2_real)
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
