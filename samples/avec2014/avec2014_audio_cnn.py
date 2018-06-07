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
"""2018/2/25 AVEC2014"""

import tensorflow as tf
from gate import context
from gate.net.custom import sensnet
from gate.data.data_factory import load_data
from gate.solver import updater
from gate.layer import l2
from gate.utils import variable
from gate.utils import filesystem
from gate.utils import string
from gate.utils.logger import logger
from samples.avec2014.utils import get_accurate_from_file


class AVEC2014_AUDIO_CNN(context.Context):

  def __init__(self, config):
    context.Context.__init__(self, config)

  def _net(self, data):
    return sensnet.SensNet(self.config.net[0], self.is_training)(data)

  def _loss(self, logit, label):
    loss = l2.loss(logit, label, self.config)
    return loss

  def _error(self, logit, label):
    mae, rmse = l2.error(logit, label, self.config)
    return mae, rmse

  @context.graph_phase_wrapper()
  def train(self):
    # load data
    image, label, path = load_data(self.config)
    # load net
    logit, _ = self._net(image)
    # compute loss
    loss = self._loss(logit, label)
    # compute error
    mae, rmse = self._error(logit, label)
    # update gradients
    global_step = tf.train.create_global_step()
    train_op = updater.default(self.config, loss, global_step)
    # add hooks
    self.add_hook(self.snapshot.init())
    self.add_hook(self.summary.init())
    self.add_hook(context.Running_Hook(
        config=self.config.log,
        step=global_step,
        keys=['loss', 'mae', 'rmse'],
        values=[loss, mae, rmse],
        # func_test=self.test,
        func_val=self.val))

    saver = tf.train.Saver(var_list=variable.all())
    with context.DefaultSession(self.hooks) as sess:
      self.snapshot.restore(sess, saver)
      while not sess.should_stop():
        sess.run(train_op)

  @context.graph_phase_wrapper()
  def val(self):
    # create a folder to save
    val_dir = filesystem.mkdir(self.config.output_dir + '/val/')
    # get data
    image, label, path = load_data(self.config)
    # get net
    logit, _ = self._net(image)
    # output to file
    info = string.concat(self.batchsize, [path, label, logit*self.data.span])
    saver = tf.train.Saver()

    # running val
    with context.DefaultSession() as sess:
      global_step = self.snapshot.restore(sess, saver)
      result_path = val_dir + '%s.txt' % global_step
      with open(result_path, 'wb') as fw:
        with context.QueueContext(sess):
          for _ in range(self.num_batch):
            _info = sess.run(info)
            [fw.write(_line + b'\r\n') for _line in _info]

      # display results on screen
      _mae, _rmse = get_accurate_from_file(result_path)
      keys = ['total sample', 'num batch', 'video_mae', 'video_rmse']
      vals = [self.total_num, self.num_batch, _mae, _rmse]
      logger.val(logger.iters(int(global_step), keys, vals))

      # write to summary
      self.summary.adds(global_step=global_step,
                        tags=['val/video_mae', 'val/video_rmse'],
                        values=[_mae, _rmse])
      return _rmse

  @context.graph_phase_wrapper()
  def test(self):
    # create a folder to save
    test_dir = filesystem.mkdir(self.config.output_dir + '/test/')
    # get data
    image, label, path = load_data(self.config)
    # get net
    logit, _ = self._net(image)
    # output to file
    info = string.concat(self.batchsize, [path, label, logit*self.data.span])
    saver = tf.train.Saver()

    # running test
    with context.DefaultSession() as sess:
      global_step = self.snapshot.restore(sess, saver)
      result_path = test_dir + '%s.txt' % global_step
      with open(result_path, 'wb') as fw:
        with context.QueueContext(sess):
          for _ in range(self.num_batch):
            _info = sess.run(info)
            [fw.write(_line + b'\r\n') for _line in _info]

      # display results on screen
      _mae, _rmse = get_accurate_from_file(result_path)
      keys = ['total sample', 'num batch', 'video_mae', 'video_rmse']
      vals = [self.total_num, self.num_batch, _mae, _rmse]
      logger.test(logger.iters(int(global_step), keys, vals))

      # write to summary
      self.summary.adds(global_step=global_step,
                        tags=['test/video_mae', 'test/video_rmse'],
                        values=[_mae, _rmse])
      return _rmse
