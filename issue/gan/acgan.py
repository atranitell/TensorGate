# -*- coding: utf-8 -*-
""" Conditional Image Synthesis With Auxiliary Classifier GANs
    https://arxiv.org/abs/1610.09585
    updated: 2017/11/22
"""
import tensorflow as tf
from core.database.factory import loads
from core.network.gans import acgan
from core.solver import updater
from core.solver import variables
from core import utils
from issue import context


class ACGAN(context.Context):
  """ ACGAN
  """

  def __init__(self, config):
    context.Context.__init__(self, config)

  def _discriminator(self, x, y, reuse=None):
    logit, net, out = acgan.discriminator(
        self.data.batchsize, x,
        self.is_train, reuse)
    return logit, net, out

  def _generator(self, y, reuse=None):
    z = tf.random_normal([self.data.batchsize,
                          self.config.net.z_dim])
    logit, net = acgan.generator(
        self.data.batchsize, z, y,
        self.is_train, reuse)
    return logit, net

  def _classifier(self, out, reuse=None):
    return acgan.classifier(out, self.data.num_classes,
                            self.is_train, reuse)

  def _loss(self, label, logit_F, logit_R, c_F, c_R):
    # discriminator loss
    D_loss_F = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(logit_F), logits=logit_F))
    D_loss_R = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(logit_R), logits=logit_R))
    D_loss = D_loss_F + D_loss_R
    # generator loss
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(logit_F), logits=logit_F))
    # info loss
    onehot_label = tf.to_float(tf.one_hot(
        label, depth=self.data.num_classes, on_value=1))
    C_loss_R = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=onehot_label, logits=c_R))
    C_loss_F = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=onehot_label, logits=c_F))
    # assemble
    C_loss = C_loss_F + C_loss_R
    return D_loss, G_loss, C_loss

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
    logit_F, net_F, out_F = self._discriminator(
        logit_G, label, reuse=False)
    # discriminate real images
    logit_R, net_R, out_R = self._discriminator(
        data, label, reuse=True)

    # classify fake
    c_F = self._classifier(out_F, reuse=False)
    # classify real
    c_R = self._classifier(out_R, reuse=True)

    # loss
    D_loss, G_loss, C_loss = self._loss(label, logit_F, logit_R, c_F, c_R)

    # t_vars = tf.trainable_variables()
    d_vars = variables.select_vars('discriminator')
    g_vars = variables.select_vars('generator')
    c_vars = variables.select_vars('classifier')

    # update
    step = tf.train.create_global_step()
    d_op = updater.default(self.config, D_loss, step, d_vars, 0)
    g_op = updater.default(self.config, G_loss, step, g_vars, 0)
    c_op = updater.default(self.config, C_loss, step, c_vars, 0)

    # assemble
    train_op = [[d_op, g_op, c_op]]
    saver = tf.train.Saver(variables.all())

    # hooks
    snapshot_hook = self.snapshot.init()
    summary_hook = self.summary.init()
    running_hook = context.Running_Hook(
        config=self.config.log,
        step=step,
        keys=['D_loss', 'G_loss', 'C_loss'],
        values=[D_loss, G_loss, C_loss],
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
    test_dir = utils.filesystem.mkdir(self.config['output_dir'] + '/test/')

    # add conditional gan
    y = [[i for i in range(10)] for i in range(10)]
    logit_G, net_G = self._generator(y, 'test')

    saver = tf.train.Saver(name='restore_all')
    with tf.Session() as sess:
      step = self.snapshot.restore(sess, saver)
      imgs = sess.run(logit_G)
      utils.image.save_images(
          images=imgs, size=[10, 10],
          path=utils.path.join_step(test_dir, step, 'png'))

    self._exit_()
