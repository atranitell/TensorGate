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
import numpy as np
from gate import context
from gate.net.vae import kinvae
from gate.layer import cosine


class KINBASE(context.Context):
  """ KINFACE BASE
  """

  def __init__(self, config):
    context.Context.__init__(self, config)

  def _encoder(self, x, reuse=None):
    return kinvae.encoder(x, self.config.net[0].z_dim, self.is_training, reuse)

  def _generator(self, z, cond, reuse=None):
    return kinvae.generator(z, cond, self.is_training, reuse)

  def _discriminator(self, x, cond, reuse=None):
    return kinvae.discriminator(x, cond, self.data.num_classes,
                                self.is_training, reuse)

  def _loss_vae(self, real, fake, mu, sigma):
    """ real: real images
      fake: generative images
      mu: the mean of encoding real images
      sigma: the std of encoding real images
    """
    marginal_likelihood = tf.reduce_sum(
        real * tf.log(fake) + (1 - real) * tf.log(1 - fake), [1, 2])
    KL_divergence = 0.5 * tf.reduce_sum(
        tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, [1])
    neg_loglikelihood = -tf.reduce_mean(marginal_likelihood)
    KL_divergence = tf.reduce_mean(KL_divergence)
    ELBO = -neg_loglikelihood - KL_divergence
    return -ELBO

  def _loss_gan(self, D_F, D_R):
    """ D_F: discriminator logit for fake image
      D_R: discriminator logit for real image
    """
    D_loss_F = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(D_F), logits=D_F))
    D_loss_R = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(D_R), logits=D_R))
    D_loss = D_loss_F + D_loss_R
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(D_F), logits=D_F))
    return D_loss, G_loss

  def _loss_metric(self, feat_x, feat_y, label):
    return cosine.loss(feat_x, feat_y, label,
                       self.batchsize,
                       is_training=self.is_training)

  def _write_feat_to_npy(self, idx, x, y, label):
    """ fast to record x1, x2, label to npy array """
    if self.phase == 'val':
      self.val_x = x if idx == 0 else np.row_stack((self.val_x, x))
      self.val_y = y if idx == 0 else np.row_stack((self.val_y, y))
      self.val_l = label if idx == 0 else np.append(self.val_l, label)
    elif self.phase == 'test':
      self.test_x = x if idx == 0 else np.row_stack((self.test_x, x))
      self.test_y = y if idx == 0 else np.row_stack((self.test_y, y))
      self.test_l = label if idx == 0 else np.append(self.test_l, label)
