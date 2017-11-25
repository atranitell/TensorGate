# -*- coding: utf-8 -*-
""" Conditional GAN
    updated: 2017/11/22
"""
import tensorflow as tf
from core.database.factory import loads
from core.network.vaes import cvae
from core.solver import updater
from core.solver import variables
from core import utils
from issue import context


class CVAE(context.Context):
  """ CVAE (D:G = 1:5)
  """

  def __init__(self, config):
    context.Context.__init__(self, config)

  def _encoder(self, x, y, reuse=None):
    mean, std = cvae.encoder(
        self.data.batchsize, x, y,
        y_dim=self.data.num_classes,
        z_dim=self.config.net.z_dim,
        is_training=self.is_train,
        reuse=reuse)
    return mean, std

  def _decoder(self, z, y, reuse=None):
    return cvae.decoder(self.data.batchsize, z, y,
                        self.is_train, reuse)

  def _loss(self, real, fake, mu, sigma):
    """
    """
    marginal_likelihood = tf.reduce_sum(
        real * tf.log(fake) + (1 - real) * tf.log(1 - fake), [1, 2])
    KL_divergence = 0.5 * tf.reduce_sum(
        tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, [1])

    neg_loglikelihood = -tf.reduce_mean(marginal_likelihood)
    KL_divergence = tf.reduce_mean(KL_divergence)

    ELBO = -neg_loglikelihood - KL_divergence
    return -ELBO

  def train(self):
    """
    """
    # set phase
    self._enter_('train')

    # get data pipeline
    real, label, path = loads(self.config)

    # encode image to a vector
    mu, sigma = self._encoder(real, label)

    # resample from the re-parameterzation
    z = mu + sigma * tf.random_normal(tf.shape(mu))

    # decoding
    fake = self._decoder(z, label, reuse=False)
    fake = tf.clip_by_value(fake, 1e-8, 1 - 1e-8)

    # loss
    loss = self._loss(real, fake, mu, sigma)

    # allocate two optimizer
    global_step = tf.train.create_global_step()
    train_op = updater.default(self.config, loss, global_step)

    # update at the same time
    saver = tf.train.Saver(var_list=variables.all())
    variables.print_trainable_list()

    # hooks
    snapshot_hook = self.snapshot.init()
    summary_hook = self.summary.init()
    running_hook = context.Running_Hook(
        config=self.config.log,
        step=global_step,
        keys=['ELBO'],
        values=[loss],
        func_test=self.test,
        func_val=None)

    # monitor session
    with tf.train.MonitoredTrainingSession(
            hooks=[running_hook, snapshot_hook, summary_hook,
                   tf.train.NanTensorHook(loss)],
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

    # for conditional gan
    y = [[i for i in range(10)] for i in range(10)]
    z = tf.random_uniform([self.data.batchsize, self.config.net.z_dim], -1, 1)
    fake = self._decoder(z, y)
    saver = tf.train.Saver()
    with tf.Session() as sess:
      step = self.snapshot.restore(sess, saver)
      imgs = sess.run(fake)
      utils.image.save_images(
          images=imgs, size=[10, 10],
          path=utils.path.join_step(test_dir, step, 'png'))

    self._exit_()
