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
from config.datasets.kinface.kinface_utils import Error
import numpy as np


class cosine_metric(context.Context):

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

  def _run(self, dst, config):
    """ common component for val and run
    dst: save record to dst filename
    config: config file to running
    """
    # get data pipeline
    data, label, path = loads(config)
    x1, x2 = tf.unstack(data, axis=1)
    path1, path2 = tf.unstack(path, axis=1)

    # total_num
    total_num = config.data.total_num
    batchsize = config.data.batchsize
    num_iter = int(total_num / batchsize)

    # get loss
    loss, loss_batch = cosine.get_loss(x1, x2, label, batchsize, False)

    # write records
    info = utils.string.concat(
        batchsize, [path1, path1, path2, label, loss_batch])

    # setting session running info
    output = [x1, x2, label, info]
    with tf.Session() as sess:
      with open('%s.txt' % dst, 'wb') as fw:
        with context.QueueContext(sess):
          for i in range(num_iter):
            _f1, _f2, _label, _info = sess.run(output)
            self._write_feat_to_npy(i, _f1, _f2, _label)
            [fw.write(_line + b'\r\n') for _line in _info]

  def test(self):
    """ """
    # SIUTATION1: validation->test*n
    # for 5 fold, every fold running a validation
    #   and 1 full-set test, 4-separate test.
    for fold in ['1', '2', '3', '4', '5']:
      self._enter_('val')
      self.config.data.entry_path = self.config.data.entry_path.replace(
          'train_1', 'train_' + fold)
      output = self.config.output_dir + '/val_' + fold
      self._run(output, self.config)
      self._exit_()

      for kin in ['', '_fs', '_fd', '_md', '_ms']:
        self._enter_('test')
        self.config.data.entry_path = self.config.data.entry_path.replace(
            'test_1', 'test' + kin + '_' + fold)
        self.config.data.total_num = 100 if kin is not '' else 400
        output = self.config.output_dir + '/test_' + fold + kin
        self._run(output, self.config)

        val_err, val_thed, test_err = Error().get_all_result(
            self.val_x, self.val_y, self.val_l,
            self.test_x, self.test_y, self.test_l, True)

        app = '%s_%s' % (kin, fold)
        keys = ['val_error' + app, 'thred' + app, 'test_error' + app]
        vals = [val_err, val_thed, test_err]
        logger.test(logger.iters(0, keys, vals))
        self._exit_()
