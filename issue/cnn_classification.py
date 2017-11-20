# -*- coding: utf-8 -*-
""" classification task for image
    updated: 2017/11/19
"""

import tensorflow as tf
from core.loss import softmax
from core.database.dataset import Dataset
from core.network.factory import network
from core.utils.logger import logger


class cnn_classification():

  def __init__(self, config):
    self.config = config
    self.phase = config['task']
    self.taskcfg = config[self.phase]

  def _net(self, data):
    logit, net = network(data, self.taskcfg, self.phase)
    return logit, net

  def _loss(self, logit, label):
    # get loss
    loss, logit = softmax.get_loss(
        logit, label, self.taskcfg['data']['num_classes'],
        self.taskcfg['data']['batchsize'])
    # get error
    error, pred = softmax.get_error(logit, label)
    return loss, error, pred

  def _updater(self):
    

  def train(self):
    """
    """
    # get data pipeline
    data, label, path = Dataset(self.taskcfg['data'], self.phase).loads()

    # setting global step
    global_step = tf.train.create_global_step()

    # get network
    logit, net = self._net(data)

    # get loss
    loss, error, pred = self._loss(logit, label)

    with tf.train.MonitoredTrainingSession() as sess:
      loss = sess.run(loss)
      print(loss)
      exit(1)

  def test(config, logger):
    pass

  def val():
    pass
