""" Machine Learning for EXTRACT FEATURE
  the inputs protocal should be:
  image-path-1
  image-path-2
  ...
  And the batchsize should be set to 1
  Reader Thread Number = 1
"""
import tensorflow as tf
from core.network.cnns.inception_resnet_v1 import inference
from core.database.factory import loads
from issue import context
import numpy as np


class EXTRACT_FEATURE(context.Context):

  def __init__(self, config):
    # if directly use input data as features
    self.direct_use_data = False
    context.Context.__init__(self, config)

  def test(self):
    self._enter_('test')

    # get data pipeline
    x, _, path = loads(self.config)

    _, end_points = inference(x, 1.0, False)
    target = end_points['PreLogitsFlatten']

    # total_num
    total_num = self.config.data.total_num
    batchsize = self.config.data.batchsize
    num_iter = int(total_num / batchsize)

    # setting session running info
    feats = []
    names = []
    saver = tf.train.Saver()
    with tf.Session() as sess:
      step = self.snapshot.restore(sess, saver)
      with context.QueueContext(sess):
        for i in range(num_iter):
          _x, _p = sess.run([target, path])
          feats.append(_x[0])
          names.append(_p)
    feat = np.array(feats)
    names = np.array(names)
    np.save(self.config.output_dir + '/feats', feat)
    np.save(self.config.output_dir + '/names', names)
    self._exit_()
