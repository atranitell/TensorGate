# Copyright 2017 The KaiJIN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Classification for CNN"""

import tensorflow as tf
from gate import context
from gate.net.net_factory import net_graph
from gate.data.data_factory import load_data
from gate.solver import updater
from gate.layer import softmax
from gate.utils import variable
from gate.utils import filesystem
from gate.utils import string
from gate.utils.logger import logger


class Classification(context.Context):

  def __init__(self, config):
    context.Context.__init__(self, config)

  def _net(self, data):
    return net_graph(data, self.config.net[0], self.phase)

  def _loss(self, logit, label):
    loss = softmax.loss(logit, label, self.config)
    return loss

  def _error(self, logit, label):
    error, pred = softmax.error(logit, label)
    return error, pred

  @context.graph_phase_wrapper()
  def train(self):
    # load data
    image, label, path = load_data(self.config)
    # load net
    logit, end_points = self._net(image)
    # compute loss
    loss = self._loss(logit, label)
    # compute error
    error, pred = self._error(logit, label)
    # update gradients
    global_step = tf.train.create_global_step()
    train_op = updater.default(self.config, loss, global_step)
    # add hooks
    self.add_hook(self.snapshot.init())
    self.add_hook(self.summary.init())
    self.add_hook(context.Running_Hook(
        config=self.config.log,
        step=global_step,
        keys=['loss', 'error'],
        values=[loss, error],
        func_test=self.test,
        func_val=None))

    saver = tf.train.Saver(var_list=variable.all())
    with context.DefaultSession(self.hooks) as sess:
      self.snapshot.restore(sess, saver)
      while not sess.should_stop():
        sess.run(train_op)

  @context.graph_phase_wrapper()
  def test(self):
    # create a folder to save
    test_dir = filesystem.mkdir(self.config.output_dir + '/test/')
    # get data
    image, label, path = load_data(self.config)
    # get net
    logit, end_points = self._net(image)
    # get loss
    loss = self._loss(logit, label)
    # get logit
    error, pred = self._error(logit, label)
    # output to file
    info = string.concat(self.batchsize, [path, label, pred])
    mean_err, mean_loss = 0, 0
    saver = tf.train.Saver()

    # running test
    with context.DefaultSession() as sess:
      global_step = self.snapshot.restore(sess, saver)
      with open(test_dir + '%s.txt' % global_step, 'wb') as fw:
        with context.QueueContext(sess):
          for _ in range(self.num_batch):
            _loss, _err, _info = sess.run([loss, error, info])
            mean_loss += _loss
            mean_err += _err
            [fw.write(_line + b'\r\n') for _line in _info]

      # statistic
      mean_loss = 1.0 * mean_loss / self.num_batch
      mean_err = 1.0 * mean_err / self.num_batch

      # display results on screen
      keys = ['total sample', 'num batch', 'loss', 'error']
      vals = [self.total_num, self.num_batch, mean_loss, mean_err]
      logger.test(logger.iters(int(global_step), keys, vals))

      # write to summary
      self.summary.adds(global_step=global_step,
                        tags=['test/error', 'test/loss'],
                        values=[mean_err, mean_loss])
      return mean_err
