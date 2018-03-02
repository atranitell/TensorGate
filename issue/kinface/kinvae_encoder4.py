# -*- coding: utf-8 -*-
""" updated: 2017/11/22
"""
import tensorflow as tf
from tensorflow.contrib import layers
from core.data.factory import loads
from core.solver import updater
from core.solver import context
from core.utils.variables import variables
from issue.kinface.kinvae_bidirect import KINVAE_BIDIRECT


class KINVAE_ENCODER4(KINVAE_BIDIRECT):
  """ """

  def __init__(self, config):
    KINVAE_BIDIRECT.__init__(self, config)

  def fc(self, x):
    return layers.fully_connected(
        x, 512,
        activation_fn=None,
        weights_initializer=tf.truncated_normal_initializer(stddev=0.01))

  def _network(self, c1_real, p2_real):
    with tf.variable_scope('net1'):
      c1_mu, c1_sigma, c1_feat = self._encoder(c1_real)
      c1_feat = self.fc(c1_feat)
    with tf.variable_scope('net2'):
      p2_mu, p2_sigma, p2_feat = self._encoder(p2_real)
      p2_feat = self.fc(p2_feat)
    return c1_feat, p2_feat

  def train(self):
    """ """
    # set phase
    self._enter_('train')

    # get data pipeline
    data, info, _ = loads(self.config)
    c1_real, p1_real, c2_real, p2_real = tf.unstack(data, axis=1)
    label, cond = tf.unstack(info, axis=1)

    # encode image to a vector
    feat_c1, feat_p2 = self._network(c1_real, p2_real)
    loss, loss_batch = self._loss_metric(feat_c1, feat_p2, label)

    # # allocate two optimizer
    global_step = tf.train.create_global_step()
    train_op = updater.default(self.config, loss, global_step)

    # update at the same time
    saver = tf.train.Saver(var_list=variables.all())

    # hooks
    self.add_hook(self.snapshot.init())
    self.add_hook(self.summary.init())
    self.add_hook(context.Running_Hook(
        config=self.config.log,
        step=global_step,
        keys=['R'],
        values=[loss],
        func_test=self.test,
        func_val=None))

    with context.DefaultSession(self.hooks) as sess:
      self.snapshot.restore(sess, saver)
      while not sess.should_stop():
        sess.run(train_op)
