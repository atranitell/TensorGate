# -*- coding: utf-8 -*-
""" updated: 2017/11/22
"""
import numpy as np
import tensorflow as tf
from core.data.factory import loads
from core.solver import updater
from core.solver import context
from core.utils.variables import variables
from core.utils.filesystem import filesystem
from core.utils.string import string
from core.utils.image import image
from issue.kinface.kinvae_bidirect import KINVAE_BIDIRECT

class KINVAE_ENCODER(KINVAE_BIDIRECT):
  """ """

  def __init__(self, config):
    KINVAE_BIDIRECT.__init__(self, config)

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