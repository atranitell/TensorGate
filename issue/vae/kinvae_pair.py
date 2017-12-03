# -*- coding: utf-8 -*-
""" Conditional GAN
    updated: 2017/11/22
"""
import tensorflow as tf
from core.database.factory import loads
from core.network.vaes import kin_vae
from core.solver import updater
from core.solver import variables
from core.loss import cosine
from core import utils
from issue import context


class KIN_VAE_PAIR(context.Context):
  """
  """

  def __init__(self, config):
    context.Context.__init__(self, config)

  def _encoder(self, x, reuse=None):
    return kin_vae.encoder(
        self.data.batchsize, x,
        z_dim=self.config.net.z_dim,
        is_training=self.is_train,
        reuse=reuse)

  def _decoder(self, z, reuse=None):
    return kin_vae.decoder(self.data.batchsize, z,
                           self.is_train, reuse)

  def _classifier(self, x, reuse=None):
    return kin_vae.classifier(x, self.data.num_classes,
                              self.is_train, reuse)

  def _discriminator(self, x, reuse=None):
    return kin_vae.discriminator(
        self.data.batchsize, x,
        self.is_train, reuse)

  def _recognitor(self, x1, x2, label):
    feat1 = kin_vae.recognitor(x1, self.is_train, 'net1')
    feat2 = kin_vae.recognitor(x2, self.is_train, 'net2')
    return cosine.get_loss(feat1, feat2, label,
                           self.data.batchsize,
                           is_training=self.is_train)

  def _loss(self, real, fake, mu, sigma):
    """
    real: real images
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
    """
    D_F: discriminator logit for fake image
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

  def train(self):
    """
    """
    # set phase
    self._enter_('train')

    # get data pipeline
    data, label, path = loads(self.config)
    real_c1, real_p1, real_p2 = tf.unstack(data, axis=1)

    # encode image to a vector
    mu, sigma = self._encoder(real_c1)

    # resample from the re-parameterzation
    z = mu + sigma * tf.random_normal(tf.shape(mu))

    # decoding
    fake = self._decoder(z)
    fake = tf.clip_by_value(fake, 1e-8, 1 - 1e-8)

    # discriminator
    D_fake, D_net_fake = self._discriminator(fake, reuse=False)
    D_real1, D_net_real1 = self._discriminator(real_p1, reuse=True)

    # recognitor
    R_loss = self._recognitor(fake, real_p2, label)

    # loss
    E_loss = self._loss(real_p1, fake, mu, sigma)
    D_loss, G_loss = self._loss_gan(D_fake, D_real1)

    loss = E_loss + D_loss + G_loss + R_loss

    # # allocate two optimizer
    global_step = tf.train.create_global_step()

    var_e = variables.select_vars('encoder')
    var_g = variables.select_vars('decoder')
    var_d = variables.select_vars('discriminator')
    var_r = variables.select_vars('recognitor')

    op1 = updater.default(self.config, loss, global_step, var_e, 0)
    op2 = updater.default(self.config, loss, None, var_g, 1)
    op3 = updater.default(self.config, loss, None, var_d, 0)
    op4 = updater.default(self.config, loss, None, var_r, 2)
    train_op = tf.group(op1, op2, op3, op4)

    # train_op = updater.default(
    #     self.config, , global_step)

    # update at the same time
    saver = tf.train.Saver(var_list=variables.all())

    # hooks
    snapshot_hook = self.snapshot.init()
    summary_hook = self.summary.init()
    running_hook = context.Running_Hook(
        config=self.config.log,
        step=global_step,
        keys=['ELBO', 'D', 'G', 'R'],
        values=[E_loss, D_loss, G_loss, R_loss],
        func_test=self.test,
        func_val=self.val)

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

  def test(self):
    """ random test a group of image
    """
    self._enter_('test')
    test_dir = utils.filesystem.mkdir(self.config.output_dir + '/test/')

    # considering output train image
    data, label, path = loads(self.config)
    real1, real2 = tf.unstack(data, axis=1)

    # encode image to a vector
    mu, sigma = self._encoder(real1)
    z = mu + sigma * tf.random_normal(tf.shape(mu))
    fake = self._decoder(z)
    fake = tf.clip_by_value(fake, 1e-8, 1 - 1e-8)
    loss = self._recognitor(fake, real2, label)

    saver = tf.train.Saver()
    num_iter = int(self.data.total_num / self.data.batchsize)
    with tf.Session() as sess:
      step = self.snapshot.restore(sess, saver)
      print(loss)
      info = utils.string.concat(self.data.batchsize, [path, label, loss])
      with open(test_dir + '%s.txt' % step, 'wb') as fw:
        # start to test
        img_dir = utils.filesystem.mkdir(test_dir + step)
        with context.QueueContext(sess):
          for i in range(num_iter):
            imgs, paths, _info = sess.run([fake, path, info])
            # utils.image.saveall(img_dir, imgs, paths)
            [fw.write(_line + b'\r\n') for _line in _info]

    self._exit_()

  def val(self):
    """ random test a group of image
    """
    self._enter_('val')
    val_dir = utils.filesystem.mkdir(self.config.output_dir + '/val/')

    # considering output train image
    data, label, path = loads(self.config)
    real1, real2 = tf.unstack(data, axis=1)

    mu, sigma = self._encoder(real1)
    z = mu + sigma * tf.random_normal(tf.shape(mu))
    fake = self._decoder(z)
    fake = tf.clip_by_value(fake, 1e-8, 1 - 1e-8)

    saver = tf.train.Saver(allow_empty=True)
    num_iter = int(self.data.total_num / self.data.batchsize)

    with tf.Session() as sess:
      step = self.snapshot.restore(sess, saver)
      with context.QueueContext(sess):
        for _ in range(num_iter):
          imgs = sess.run(fake)
          utils.image.save_images(
              images=imgs, size=[10, 10],
              path=utils.path.join_step(val_dir, step, 'png'))

    self._exit_()