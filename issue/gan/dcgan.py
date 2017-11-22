# -*- coding: utf-8 -*-
""" GAN for normal image
    updated: 2017/11/22
"""

import os
import tensorflow as tf

from core.loss import softmax
from core.database.dataset import Dataset
from core.network.nets import dcgan
from core.solver.updater import Updater
from core.solver.snapshot import Snapshot
from core.solver.summary import Summary

from core.utils import filesystem
from core.utils import string
from core.utils.logger import logger
from core.utils.context import QueueContext

from issue.running_hook import Running_Hook
from issue.gan import utils


class DCGAN():
  """ DCGAN 
  net:
    image_fake, net_fake = Generator(random_vector)
    logit_fake = Discriminator(image_fake)
    logit_real = Discriminator(iamge_real)
  loss:
    d_real_loss = cross_entropy(logit_real, 1)
    d_fake_loss = corss_entropy(logit_fake, 0)
    d_loss = d_real_loss + d_fake_loss
    g_loss = cross_entropy(d_loss, 1

    optimize(d_loss)
    optimize(g_loss)

  """

  def __init__(self, config):
    self.config = config
    self.summary = Summary(self.config['log'], config['output_dir'])
    self.snapshot = Snapshot(self.config['log'], config['output_dir'])
    # current work env
    self.taskcfg = None

  def _enter_(self, phase):
    """ task enter
    """
    self.pre_taskcfg = self.taskcfg
    self.taskcfg = self.config[phase]
    self.datacfg = self.taskcfg['data']

  def _exit_(self):
    """ task exit
    """
    self.taskcfg = self.pre_taskcfg
    self.datacfg = self.taskcfg['data']

  def _discriminator(self, X, phase, reuse=None):
    is_training = True if phase == 'train' else False
    logit, net = dcgan.discriminator(X, depth=64,
                                     is_training=is_training,
                                     reuse=reuse,
                                     fused_batch_norm=True)
    return logit, net

  def _generator(self, batchsize, z_dim, phase):
    """ from z distribution to sample a random vector
    """
    z = tf.random_normal([batchsize, z_dim])
    is_training = True if phase == 'train' else False
    logit, net = dcgan.generator(z, depth=64, final_size=32,
                                 is_training=is_training,
                                 fused_batch_norm=True)
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
    logit_G, net_G = self._generator(self.datacfg['batchsize'], 100, 'train')
    # discriminate fake images
    logit_F, net_F = self._discriminator(logit_G, 'train', reuse=False)
    # discriminate real images
    logit_R, net_R = self._discriminator(data, 'train', reuse=True)

    # loss
    D_loss, G_loss = self._loss(logit_F, logit_R)

    # acquire update list
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'Discriminator' in var.name]
    g_vars = [var for var in t_vars if 'Generator' in var.name]

    # update
    global_step = tf.train.create_global_step()

    # pay attention
    # there global_step just running once
    d_optim = tf.train.AdamOptimizer(0.00002).minimize(
        loss=D_loss, global_step=global_step, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(0.00002).minimize(
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
    logit_G, net_G = self._generator(self.datacfg['batchsize'], 100, 'test')
    saver = tf.train.Saver(name='restore_all')
    with tf.Session() as sess:
      global_step = self.snapshot.restore(sess, saver)
      imgs = sess.run(logit_G)
      img_path = os.path.join(test_dir, '{:08d}.png'.format(int(global_step)))
      utils.save_images(imgs, [8, 8], img_path)
    self._exit_()
