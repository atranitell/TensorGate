# -*- coding: utf-8 -*-
""" Context manager tools
    updated: 2017/11/19
"""

import tensorflow as tf


class QueueContext():
  """ For managing the data reader queue.
  """

  def __init__(self, sess):
    self.sess = sess

  def __enter__(self):
    self.coord = tf.train.Coordinator()
    self.threads = []
    for queuerunner in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
      self.threads.extend(queuerunner.create_threads(
          self.sess, coord=self.coord, daemon=True, start=True))

  def __exit__(self, *unused):
    self.coord.request_stop()
    self.coord.join(self.threads, stop_grace_period_secs=10)
