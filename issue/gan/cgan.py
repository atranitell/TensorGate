# -*- coding: utf-8 -*-
""" Conditional GAN
    updated: 2017/11/22
"""

import os
import tensorflow as tf

from core.loss import softmax
from core.database.factory import loads
from core.network.gans import cgan
from core.solver import updater
from core.solver import variables
from core.solver.snapshot import Snapshot
from core.solver.summary import Summary

from core.utils import filesystem
from core.utils import string
from core.utils.logger import logger
from core.utils.context import QueueContext

from issue.running_hook import Running_Hook
from issue.gan import utils


class CGAN():
  """ CGAN
  """

  def __init__(self, config):
    self.config = config
    self.phase = config.phase
    self.data = config.data
    self.summary = Summary(config)
    self.snapshot = Snapshot(config)

  def _enter_(self, phase):
    self.prephase = self.phase
    self.phase = phase
    self.config.set_phase(phase)
    self.data = self.config.data

  def _exit_(self):
    self.phase = self.prephase
    self.config.set_phase(self.phase)
    self.data = self.config.data

  def _discriminator(self, x, y, reuse=None):
    is_training = True if self.phase == 'train' else False
    logit, net = cgan.discriminator(
        self.data.batchsize, x, y,
        self.data.num_classes,
        is_training, reuse)
    return logit, net

  def _generator(self, y, reuse=None):
    """ from z distribution to sample a random vector
    """
    z = tf.random_normal([self.data.batchsize, self.config.net.z_dim])
    is_training = True if self.phase == 'train' else False
    logit, net = cgan.generator(
        self.data.batchsize, z, y,
        is_training, reuse)
    return logit, net

  def _loss(self, logit_F, logit_R):
    """
    """
    D_loss_F = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(logit_F), logits=logit_F))
    D_loss_R = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(logit_R), logits=logit_R))
    D_loss = D_loss_F + D_loss_R

    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(logit_F), logits=logit_F))
    return D_loss, G_loss

  def train(self):
    """
    """
    # set phase
    self._enter_('train')

    # get data pipeline
    data, label, path = loads(self.config)
    # generate fake images
    logit_G, net_G = self._generator(label)
    # discriminate fake images
    logit_F, net_F = self._discriminator(logit_G, label, reuse=False)
    # discriminate real images
    logit_R, net_R = self._discriminator(data, label, reuse=True)

    # loss
    D_loss, G_loss = self._loss(logit_F, logit_R)

    # acquire update list
    d_vars = variables.select_vars('discriminator')
    g_vars = variables.select_vars('generator')

    # allocate two optimizer
    global_step = tf.train.create_global_step()
    d_optim = updater.default(
        self.config, D_loss, global_step, d_vars, index=0)
    g_optim = updater.default(
        self.config, G_loss, None, g_vars, index=1)

    # update at the same time
    train_op = [[d_optim, g_optim]]
    saver = tf.train.Saver(var_list=variables.all())
    variables.print_trainable_list()

    # hooks
    snapshot_hook = self.snapshot.init()
    summary_hook = self.summary.init()
    running_hook = Running_Hook(
        config=self.config.log,
        step=global_step,
        keys=['D_loss', 'G_loss'],
        values=[D_loss, G_loss],
        func_test=self.test,
        func_val=None)

    # monitor session
    with tf.train.MonitoredTrainingSession(
            hooks=[running_hook, snapshot_hook, summary_hook,
                   tf.train.NanTensorHook(G_loss),
                   tf.train.NanTensorHook(D_loss)],
            save_checkpoint_secs=None,
            save_summaries_steps=None) as sess:

      # restore model
      self.snapshot.restore(sess, saver)

      # running
      while not sess.should_stop():
        sess.run(train_op)

  def test(self):
    """ random test a group of image
    """
    self._enter_('test')
    test_dir = filesystem.mkdir(self.config.output_dir + '/test/')

    # for conditional gan
    y = [[i for i in range(10)] for i in range(10)]
    logit_G, net_G = self._generator(y)
    saver = tf.train.Saver()
    with tf.Session() as sess:
      global_step = self.snapshot.restore(sess, saver)
      imgs = sess.run(logit_G)
      img_path = os.path.join(test_dir, '{:08d}.png'.format(int(global_step)))
      utils.save_images(imgs, [10, 10], img_path)
    self._exit_()
