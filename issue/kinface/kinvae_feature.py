""" Machine Learning for Kinface
  the model to directly use features from .npy file
  to acquire a similarity metric.
  It will not include any train parts.

  For most cosine metric algorithms,
  it will need find a best margin in validation test.
  And then use margin to divide the test dataset.
"""
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from core.data.factory import loads
from core.loss import cosine
from core.solver import updater
from core.solver import context
from core.utils import similarity
from core.utils.variables import variables
from core.utils.filesystem import filesystem
from core.utils.logger import logger
from core.utils.string import string


class KINVAE_FEATURE(context.Context):

  def __init__(self, config):
    # if directly use input data as features
    self.direct_use_data = False
    context.Context.__init__(self, config)

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

  def linear(self, x, output_size, activation_fn, scope="linear", reuse=None):
    return layers.fully_connected(
        inputs=x,
        num_outputs=output_size,
        activation_fn=activation_fn,
        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
        reuse=reuse,
        scope=scope)

  def mlp(self, x, reuse=None):
    net = self.linear(x, 1024, tf.nn.relu, 'fc1', reuse)
    net = self.linear(x, 512, tf.nn.relu, 'fc2', reuse)
    net = self.linear(x, 256, None, 'logit', reuse)
    return net

  def _network(self, feat1, feat2):
    """
    """
    feat1 = self.mlp(feat1)
    feat2 = self.mlp(feat2, True)
    return feat1, feat2

  def train(self):
    """
    """
    """ """
    # set phase
    self._enter_('train')

    # get data pipeline
    data, label, path = loads(self.config)
    x1, x2 = tf.unstack(data, axis=1)
    path1, path2 = tf.unstack(path, axis=1)

    # encode image to a vector
    x1, x2 = self._network(x1, x2)
    loss, loss_batch = cosine.get_loss(x1, x2, label, self.batchsize, True)

    # # allocate two optimizer
    global_step = tf.train.create_global_step()
    train_op = updater.default(self.config, loss, global_step)

    # update at the same time
    saver = tf.train.Saver(var_list=variables.all())

    # hooks
    self.add_hook(self.snapshot.init())
    self.add_hook(self.summary.init())
    self.add_hook(context.Running_Hook(
        config=self.config.log,
        step=global_step,
        keys=['R'],
        values=[loss],
        func_test=self.test,
        func_val=None))

    with context.DefaultSession(self.hooks) as sess:
      self.snapshot.restore(sess, saver)
      while not sess.should_stop():
        sess.run(train_op)

  def _val_or_test(self, dst):
    """ common component for val and run
    dst: save record to dst filename
    config: config file to running
    """
    # get data pipeline
    data, label, path = loads(self.config)
    x1, x2 = tf.unstack(data, axis=1)
    path1, path2 = tf.unstack(path, axis=1)

    # get loss
    x1, x2 = self._network(x1, x2)
    loss, loss_batch = cosine.get_loss(x1, x2, label, self.batchsize, False)
    # write records
    info = string.concat(
        self.batchsize, [path1, path1, path2, label, loss_batch])

    # setting session running info
    output = [x1, x2, label, info]
    saver = tf.train.Saver()
    with context.DefaultSession() as sess:
      step = self.snapshot.restore(sess, saver)
      fw = open('%s/%s.txt' % (dst, step), 'wb')
      with context.QueueContext(sess):
        for i in range(self.epoch_iter):
          _f1, _f2, _label, _info = sess.run(output)
          self._write_feat_to_npy(i, _f1, _f2, _label)
          [fw.write(_line + b'\r\n') for _line in _info]
      fw.close()
      return step

  def test(self):
    """ """
    with tf.Graph().as_default():
      self._enter_('val')
      dst_dir = filesystem.mkdir(self.config.output_dir + '/val/')
      self._val_or_test(dst_dir)
      self._exit_()

    with tf.Graph().as_default():
      self._enter_('test')
      dst_dir = filesystem.mkdir(self.config.output_dir + '/test/')
      step = self._val_or_test(dst_dir)

      val_err, val_thed, test_err = similarity.get_all_result(
          self.val_x, self.val_y, self.val_l,
          self.test_x, self.test_y, self.test_l, False)

      keys = ['val_error', 'thred', 'test_error']
      vals = [val_err, val_thed, test_err]
      logger.test(logger.iters(int(step) - 1, keys, vals))
      self._exit_()
