# -*- coding: utf-8 -*-
""" pairwise task for image
    updated: 2017/11/19
"""
import tensorflow as tf
from core.database.factory import loads
from core.network.factory import network
from core.loss import cosine
from core.solver import updater
from core.solver import variables
from core import utils
from core.utils.logger import logger
from core.utils.profiler import Profiler
from issue import context


class pairwise(context.Context):

  def __init__(self, config):
    context.Context.__init__(self, config)

  def _net(self, x1, x2, label):
    logit, net1 = network(x1, self.config, self.phase, 'net1')
    logit, net2 = network(x2, self.config, self.phase, 'net2')
    feat1 = net1['global_pool']
    feat2 = net2['global_pool']
    return cosine.get_loss(feat1, feat2, label, self.data.batchsize, self.is_train)

  def train(self):
    """
    """
    # set phase
    self._enter_('train')

    # get data pipeline
    data, label, path = loads(self.config)
    x1, x2 = tf.unstack(data, axis=1)

    # update
    loss = self._net(x1, x2, label)
    global_step = tf.train.create_global_step()
    train_op = updater.default(self.config, loss, global_step)

    # for storage
    saver = tf.train.Saver(var_list=variables.all())
    variables.print_trainable_list()

    # hooks
    snapshot_hook = self.snapshot.init()
    summary_hook = self.summary.init()
    running_hook = context.Running_Hook(
        config=self.config.log,
        step=global_step,
        keys=['loss'],
        values=[loss],
        func_test=self.test,
        func_val=None)

    # monitor session
    with tf.train.MonitoredTrainingSession(
            hooks=[running_hook, snapshot_hook, summary_hook,
                   tf.train.NanTensorHook(loss)],
            save_checkpoint_secs=None,
            save_summaries_steps=None) as sess:

      # restore model: if checkpoint does not exited, do nothing
      self.snapshot.restore(sess, saver)

      # Profile
      # Profiler.time_memory(self.config['output_dir'], sess, train_op)

      while not sess.should_stop():
        sess.run(train_op)

  def test(self):
    """
    """
    # save current context
    self._enter_('test')

    # create a folder to save
    test_dir = utils.filesystem.mkdir(self.config.output_dir + '/test/')
    # get data pipeline
    data, label, path = loads(self.config)
    x1, x2 = tf.unstack(data, axis=1)
    # total_num
    total_num = self.data.total_num
    batchsize = self.data.batchsize
    # get loss
    loss = self._net(x1, x2, label)
    # get saver
    saver = tf.train.Saver()
    with tf.Session() as sess:
      # get latest checkpoint
      global_step = self.snapshot.restore(sess, saver)
      # output to file
      info = utils.string.concat(batchsize, [path, label, loss])
      with open(test_dir + '%s.txt' % global_step, 'wb') as fw:
        with context.QueueContext(sess):
          # Initial some variables
          num_iter = int(total_num / batchsize)
          mean_loss = 0
          for _ in range(num_iter):
            # running session to acuqire value
            _loss, _info = sess.run([loss, info])
            mean_loss += _loss
            # save tensor info to text file
            [fw.write(_line + b'\r\n') for _line in _info]
          # statistic
          mean_loss = 1.0 * mean_loss / num_iter
      # display results on screen
      keys = ['total sample', 'num batch', 'loss']
      vals = [total_num, num_iter, mean_loss]
      logger.test(logger.iters(int(global_step), keys, vals))
      # write to summary
      self.summary.adds(global_step=global_step,
                        tags=['test/loss'],
                        values=[mean_loss])

      self._exit_()
      return mean_loss

  def val(self):
    pass
