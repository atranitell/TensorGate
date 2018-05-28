# Copyright 2017 The KaiJIN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Kinface Base"""

import tensorflow as tf
import numpy as np
from gate import context
from gate.net.vae import kinvae
from gate.data.data_factory import load_data
from gate.layer import cosine
from gate.layer import similarity
from gate.utils import filesystem
from gate.utils import string
from gate.utils.logger import logger


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

  def _inference(self, output_dir):
    """ COMMON FOR TRAIN AND VAL """
    # considering output train image
    data, info, path = load_data(self.config)
    c1_real, p1_real, c2_real, p2_real = tf.unstack(data, axis=1)
    c1_path, p1_path, c2_path, p2_path = tf.unstack(path, axis=1)
    label, cond = tf.unstack(info, axis=1)

    # encode image to a vector
    feat_c1, feat_p2 = self._net(c1_real, p2_real)
    R_loss, loss = self._loss_metric(feat_c1, feat_p2, None)

    saver = tf.train.Saver()

    with context.DefaultSession() as sess:
      step = self.snapshot.restore(sess, saver)
      info = string.concat(
          self.batchsize,
          [c1_path, p1_path, c2_path, p2_path, label, loss])

      output = [info, feat_c1, feat_p2, label]
      fw = open(output_dir + '%s.txt' % step, 'wb')

      with context.QueueContext(sess):
        for i in range(self.num_batch):
          _info, _x, _y, _label = sess.run(output)
          self._write_feat_to_npy(i, _x, _y, _label)
          [fw.write(_line + b'\r\n') for _line in _info]
          c1_zs = _x if i == 0 else np.row_stack((c1_zs, _x))
          p2_zs = _y if i == 0 else np.row_stack((p2_zs, _y))

      fw.close()
      np.save(output_dir + '/' + step + '_c1.npy', c1_zs)
      np.save(output_dir + '/' + step + '_p2.npy', p2_zs)

      return step
      
  def test(self):
    """ we need acquire threshold from validation first """
    with tf.Graph().as_default():
      self._enter_('val')
      _val_dir = filesystem.mkdir(self.config.output_dir + '/val/')
      _step = self._inference(_val_dir)
      self._exit_()

    with tf.Graph().as_default():
      self._enter_('test')
      _tst_dir = filesystem.mkdir(self.config.output_dir + '/test/')
      _step = self._inference(_tst_dir)
      val_err, val_thed, test_err = similarity.get_all_result(
          self.val_x, self.val_y, self.val_l,
          self.test_x, self.test_y, self.test_l, False)
      keys = ['val_error', 'thred', 'test_error']
      vals = [val_err, val_thed, test_err]
      logger.test(logger.iters(int(_step) - 1, keys, vals))
      self._exit_()
