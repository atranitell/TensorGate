# -*- coding: utf-8 -*-
""" classification task for image
    updated: 2017/11/19
"""

import os
import tensorflow as tf
from core.loss import softmax
from core.database.dataset import Dataset
from core.network.factory import network
from core.solver.updater import Updater
from core.solver.snapshot import Snapshot
from core.utils.logger import logger
from issue.running_hook import Running_Hook


class cnn_classification():

  def __init__(self, config):
    self.config = config
    self.phase = config['task']
    self.taskcfg = config[self.phase]
    self.summary = None

  def _net(self, data):
    logit, net = network(data, self.config, self.phase)
    return logit, net

  def _loss(self, logit, label):
    # get loss
    loss, logit = softmax.get_loss(
        logit, label, self.taskcfg['data']['num_classes'],
        self.taskcfg['data']['batchsize'])

    # get error
    error, pred = softmax.get_error(logit, label)
    return loss, error, pred

  def as_batch(self, tensor, batchsize):
    """ convert tensor to string type
    """
    return tf.as_string(tf.reshape(tensor, shape=[batchsize]))

  def train(self):
    """
    """
    # set phase
    self.phase = 'train'
    self.taskcfg = self.config[self.phase]

    # get data pipeline
    data, label, path = Dataset(self.taskcfg['data'], self.phase).loads()

    # get network
    logit, net = self._net(data)

    # get loss
    loss, error, pred = self._loss(logit, label)

    # update
    with tf.name_scope('updater'):
      updater = Updater()
      updater.init_default_updater(self.taskcfg, loss)
      lr = updater.get_learning_rate()
      train_op = updater.get_train_op()
      global_step = updater.get_global_step()
      restore_saver = updater.get_variables_saver()

    # checkpoint
    with tf.name_scope('checkpoint'):
      snapshot = Snapshot(self.config['log'])
      chkp_hook = snapshot.get_chkp_hook(self.config['name'],
                                         self.config['output_dir'])
      summary_hook = snapshot.get_summary_hook(self.config['output_dir'])
      self.summary = snapshot.get_summary(self.config['output_dir'])

    # hooks
    running_hook = Running_Hook(self.config['log'], global_step,
                                ['loss', 'error'], [loss, error],
                                func_test=self.test,
                                func_val=None)

    with tf.train.MonitoredTrainingSession(
            hooks=[running_hook, chkp_hook, summary_hook,
                   tf.train.NanTensorHook(loss)],
            save_checkpoint_secs=None,
            save_summaries_steps=None) as sess:

      if 'restore' in self.taskcfg and self.taskcfg['restore']:
        snapshot.restore(sess, self.config['output_dir'], restore_saver)

      while not sess.should_stop():
        sess.run(train_op)

  def test(self):
    """
    """
    # store cur phase
    cur_phase = self.phase
    cur_taskcfg = self.taskcfg
    self.phase = 'test'
    self.taskcfg = self.config[self.phase]

    # create a folder to save
    output_dir = self.config['output_dir'] + '/test/'
    if not os.path.exists(output_dir):
      os.mkdir(output_dir)

    # get data pipeline
    data, label, path = Dataset(self.taskcfg['data'], self.phase).loads()

    # get network
    logit, net = self._net(data)

    # get loss
    loss, error, pred = self._loss(logit, label)

    # get saver
    saver = tf.train.Saver(name='restore_all')

    with tf.Session() as sess:
      # get latest checkpoint
      snapshot = Snapshot(self.config['log'])
      global_step = snapshot.restore(sess, self.config['output_dir'], saver)

      # Initial some variables
      num_iter = int(self.taskcfg['data']['total_num'] /
                     self.taskcfg['data']['batchsize'])

      mean_err, mean_loss = 0, 0

      # start queue from runner
      coord = tf.train.Coordinator()
      threads = []
      for queuerunner in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(queuerunner.create_threads(
            sess, coord=coord, daemon=True, start=True))

      # output to file
      tab = tf.constant(' ', shape=[self.taskcfg['data']['batchsize']])
      label_str = self.as_batch(label, self.taskcfg['data']['batchsize'])
      pred_str = self.as_batch(pred, self.taskcfg['data']['batchsize'])
      info_batch = path + tab + label_str + tab + pred_str

      info_path = output_dir + '%s.txt' % global_step
      info_fp = open(info_path, 'wb')

      # start to run
      tf.train.start_queue_runners(sess=sess)
      for _ in range(num_iter):
        # running session to acuqire value
        _loss, _err, _info = sess.run([loss, error, info_batch])
        mean_loss += _loss
        mean_err += _err
        # save tensor info to text file
        for _line in _info:
          info_fp.write(_line + b'\r\n')

      # stop
      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)
      info_fp.close()

      # statistic
      mean_loss = 1.0 * mean_loss / num_iter
      mean_err = 1.0 * mean_err / num_iter

      # output result
      keys = ['total sample', 'num batch', 'loss', 'err']
      vals = [self.taskcfg['data']['total_num'],
              num_iter, mean_loss, mean_err]
      logger.test(logger.iters(int(global_step), keys, vals))

      # write to summary
      if self.summary is not None:
        summary = tf.Summary()
        summary.value.add(
            tag=self.phase + '/iter', simple_value=int(global_step))
        summary.value.add(tag=self.phase + '/err', simple_value=mean_err)
        summary.value.add(tag=self.phase + '/loss', simple_value=mean_loss)
        self.summary.add_summary(summary, global_step)

      self.phase = cur_phase
      self.taskcfg = cur_taskcfg
      return mean_err

  def val(self):
    pass
