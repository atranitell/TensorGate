""" Machine Learning for Kinface
  the model to directly use features from .npy file
  to acquire a similarity metric.
  It will not include any train parts.

  For most cosine metric algorithms,
  it will need find a best margin in validation test.
  And then use margin to divide the test dataset.
"""
import tensorflow as tf
from core.database.factory import loads
from core.loss import cosine
from core import utils
from core.utils.logger import logger
from issue import context
import numpy as np


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
    loss, loss_batch = cosine.get_loss(x1, x2, label, self.batchsize, False)
    # write records
    info = utils.string.concat(
        self.batchsize, [path1, path1, path2, label, loss_batch])

    # setting session running info
    output = [x1, x2, label, info]
    fw = open('%s.txt' % dst, 'wb')
    with tf.Session() as sess:
      with context.QueueContext(sess):
        for i in range(self.epoch_iter):
          _f1, _f2, _label, _info = sess.run(output)
          self._write_feat_to_npy(i, _f1, _f2, _label)
          [fw.write(_line + b'\r\n') for _line in _info]
      fw.close()

  def test(self):
    """ """
    self._enter_('val')
    dst_dir = utils.filesystem.mkdir(self.config.output_dir + '/val/')
    self._val_or_test(dst_dir)
    self._exit_()

    self._enter_('test')
    dst_dir = utils.filesystem.mkdir(self.config.output_dir + '/test/')
    self._val_or_test(dst_dir)

    val_err, val_thed, test_err = utils.similarity.get_all_result(
        self.val_x, self.val_y, self.val_l,
        self.test_x, self.test_y, self.test_l, True)

    keys = ['val_error', 'thred', 'test_error']
    vals = [val_err, val_thed, test_err]
    logger.test(logger.iters(0, keys, vals))
    self._exit_()
