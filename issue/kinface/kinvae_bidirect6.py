# -*- coding: utf-8 -*-
""" Conditional GAN
    updated: 2017/11/22
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


class KINVAE_BIDIRECT6(KINVAE_BIDIRECT):
  """ 1E - 1G
      no cosine loss
      no discrimintor error
  """

  def __init__(self, config):
    KINVAE_BIDIRECT.__init__(self, config)
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

    loss = E_loss

    # # allocate two optimizer
    global_step = tf.train.create_global_step()

    var_e = variables.select_vars('encoder')
    var_g = variables.select_vars('generator')

    op1 = updater.default(self.config, loss, global_step, var_e, 0)
    op2 = updater.default(self.config, loss, None, var_g, 1)
    train_op = tf.group(op1, op2)

    # update at the same time
    saver = tf.train.Saver(var_list=variables.all())

    # hooks
    self.add_hook(self.snapshot.init())
    self.add_hook(self.summary.init())
    self.add_hook(context.Running_Hook(
        config=self.config.log,
        step=global_step,
        keys=['E'],
        values=[E_loss],
        func_test=self.test,
        func_val=None))

    with context.DefaultSession(self.hooks) as sess:
      self.snapshot.restore(sess, saver)
      while not sess.should_stop():
        sess.run(train_op)

  def _val_or_test(self, dstdir):
    """ COMMON FOR TRAIN AND VAL """
    # considering output train image
    data, info, path = loads(self.config)
    c1_real, p1_real, c2_real, p2_real = tf.unstack(data, axis=1)
    c1_path, p1_path, c2_path, p2_path = tf.unstack(path, axis=1)
    label, cond = tf.unstack(info, axis=1)

    # encode image to a vector
    c1_mu, c1_sigma, feat_c1 = self._encoder(c1_real)
    p2_mu, p2_sigma, feat_p2 = self._encoder(p2_real, True)

    c1_z = c1_mu + c1_sigma * tf.random_normal(tf.shape(c1_mu))
    c1_z = self._generator(c1_z, cond)
    c1_fake = tf.clip_by_value(c1_z, 1e-8, 1 - 1e-8)

    p2_z = p2_mu + p2_sigma * tf.random_normal(tf.shape(p2_mu))
    p2_z = self._generator(p2_z, cond, True)
    p2_fake = tf.clip_by_value(p2_z, 1e-8, 1 - 1e-8)

    R_loss, loss = self._loss_metric(feat_c1, feat_p2, None)

    saver = tf.train.Saver()
    c1_zs, p2_zs = 0, 0

    with context.DefaultSession() as sess:
      step = self.snapshot.restore(sess, saver)
      filesystem.mkdir(dstdir + '/' + step + '_c1')
      filesystem.mkdir(dstdir + '/' + step + '_p2')

      info = string.concat(
          self.data.batchsize,
          [c1_path, p1_path, c2_path, p2_path, label, loss])

      output = [c1_fake, p2_fake, c1_path, p2_path,
                c1_real, c2_real, p1_real, p2_real,
                info, feat_c1, feat_p2, label]

      fw = open(dstdir + '%s.txt' % step, 'wb')
      with context.QueueContext(sess):
        for i in range(self.epoch_iter):
          _c1, _p2, _c1p, _p2p, _c1r, _c2r, _p1r, _p2r, _info, _x, _y, _label = sess.run(
              output)
          self._write_feat_to_npy(i, _x, _y, _label)
          [fw.write(_line + b'\r\n') for _line in _info]

          if self.is_save_all_images:
            image.save_multi(dstdir + '/' + step + '_c1', _c1, _c1p)
            image.save_multi(dstdir + '/' + step + '_p2', _p2, _p2p)

          if self.is_save_batch_images:
            image.save_batchs_to_one(
                imgs_list=[_c1, _p2, _c1r, _c2r, _p1r, _p2r],
                batchsize=self.data.batchsize, dst=dstdir, step=step,
                name_list=['_c1', '_p2', '_c1r', '_c2r', '_p1r', '_p2r'])

          if self.is_save_feats:
            c1_zs = _x if type(c1_zs) == int else np.row_stack((c1_zs, _x))
            p2_zs = _y if type(p2_zs) == int else np.row_stack((p2_zs, _y))

      fw.close()
      if self.is_save_feats:
        np.save(dstdir + '/' + step + '_c1.npy', c1_zs)
        np.save(dstdir + '/' + step + '_p2.npy', p2_zs)

      return step
