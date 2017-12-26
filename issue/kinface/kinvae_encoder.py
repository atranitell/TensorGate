# -*- coding: utf-8 -*-
""" updated: 2017/11/22
"""
import tensorflow as tf
from core.database.factory import loads
from core.solver import updater
from core.solver import variables
from issue import context
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
    snapshot_hook = self.snapshot.init()
    summary_hook = self.summary.init()
    running_hook = context.Running_Hook(
        config=self.config.log,
        step=global_step,
        keys=['R'],
        values=[loss],
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