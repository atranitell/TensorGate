# -*- coding: utf-8 -*-
""" Conditional GAN
    updated: 2017/11/22
"""

import os
import tensorflow as tf

from core.loss import softmax
from core.database.dataset import Dataset
from core.network.gans import cgan
from core.solver.updater import Updater
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
    self.summary = Summary(self.config['log'], config['output_dir'])
    self.snapshot = Snapshot(self.config['log'], config['output_dir'])
    # current work env
    self.taskcfg = None
    self.z_dim = 100

  def _enter_(self, phase):
    """ task enter
    """
    self.pre_taskcfg = self.taskcfg
    self.taskcfg = self.config[phase]
    self.datacfg = self.taskcfg['data']
    self.batchsize = self.datacfg['batchsize']
    self.num_classes = self.datacfg['num_classes']

  def _exit_(self):
    """ task exit
    """
    self.taskcfg = self.pre_taskcfg
    self.datacfg = self.taskcfg['data']
    self.batchsize = self.datacfg['batchsize']
    self.num_classes = self.datacfg['num_classes']

  def _discriminator(self, x, y, phase, reuse=None):
    is_training = True if phase == 'train' else False
    logit, net = cgan.discriminator(
        self.batchsize, x, y, self.num_classes, is_training, reuse)
    return logit, net

  def _generator(self, y, phase, reuse=None):
    """ from z distribution to sample a random vector
    """
    z = tf.random_normal([self.batchsize, self.z_dim])
    is_training = True if phase == 'train' else False
    logit, net = cgan.generator(self.batchsize, z, y, is_training, reuse)
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
    data, label, path = Dataset(self.datacfg, 'train').loads()
    # generate fake images
    logit_G, net_G = self._generator(label, 'train')
    # discriminate fake images
    logit_F, net_F = self._discriminator(logit_G, label, 'train', reuse=False)
    # discriminate real images
    logit_R, net_R = self._discriminator(data, label, 'train', reuse=True)

    # loss
    D_loss, G_loss = self._loss(logit_F, logit_R)

    # acquire update list
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'discriminator' in var.name]
    g_vars = [var for var in t_vars if 'generator' in var.name]

    # update
    global_step = tf.train.create_global_step()

    # pay attention
    # there global_step just running once
    d_optim = tf.train.AdamOptimizer(0.0001).minimize(
        loss=D_loss, global_step=global_step, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(0.0001).minimize(
        loss=G_loss, var_list=g_vars)

    train_op = [[d_optim, g_optim]]
    restore_saver = tf.train.Saver(var_list=tf.trainable_variables())

    # hooks
    snapshot_hook = self.snapshot.init()
    summary_hook = self.summary.init()
    running_hook = Running_Hook(
        config=self.config['log'],
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
      if 'restore' in self.taskcfg and self.taskcfg['restore']:
        self.snapshot.restore(sess, restore_saver)

      # running
      while not sess.should_stop():
        sess.run(train_op)

  def test(self):
    """ random test a group of image
    """
    self._enter_('test')
    test_dir = filesystem.mkdir(self.config['output_dir'] + '/test/')

    # for conditional gan
    y = [[i for i in range(10)] for i in range(10)]
    logit_G, net_G = self._generator(y, 'test')
    saver = tf.train.Saver(name='restore_all')
    with tf.Session() as sess:
      global_step = self.snapshot.restore(sess, saver)
      imgs = sess.run(logit_G)
      img_path = os.path.join(test_dir, '{:08d}.png'.format(int(global_step)))
      utils.save_images(imgs, [10, 10], img_path)
    self._exit_()
