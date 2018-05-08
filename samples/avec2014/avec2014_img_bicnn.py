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
"""2018/2/25 AVEC2014

For noraml situation, we accept two kind of inputs like (RGB + OPTICAL FLOW)
The first input use net[0].
The second input use net[1].

we set three distict concat method.
1) Normal Concat
  RGB -> RGB_feature |
                     | -> final_logit
  OPT -> OPT_feature | 

2) RGB as main branch (inputs[0])
  OPT -> OPT_feature -> OPT_logit
              ↓             ↓     OPT_feature |
  RGB ->------------------------> RGB_feature | -> final_logit
                                  OPT_logit   |

3) Optical Flow as main branch (inputs[1])
  RGB -> RGB_feature -> RGB_logit
              ↓             ↓     RGB_feature |
  OPT ->------------------------> OPT_feature | -> final_logit
                                  RGB_logit   |
"""

import tensorflow as tf
from gate import context
from gate.data.data_factory import load_data
from gate.net.net_factory import net_graph
from gate.solver import updater
from gate.layer import l2
from gate.utils import variable
from gate.utils import filesystem
from gate.utils import string
from gate.utils.logger import logger
from gate.utils.heatmap import HeatMap
from samples.avec2014.utils import get_accurate_from_file


class AVEC2014_IMG_BICNN(context.Context):

  def __init__(self, config):
    context.Context.__init__(self, config)

  def _net(self, data):
    # if using deep-fused network
    if self.config.target.startswith('avec2014.img.bicnn.shared'):
      return net_graph(data, self.config.net[0], self.phase)
    # using normal concat method
    data = tf.unstack(data, axis=1)
    # get network
    logit0, net0 = net_graph(data[0], self.config.net[0], self.phase, 'RBG')
    logit1, net1 = net_graph(data[1], self.config.net[1], self.phase, 'OPT')
    # get features
    net0 = tf.squeeze(net0['global_pool'], [1, 2])
    net1 = tf.squeeze(net1['global_pool'], [1, 2])
    # different fuse method
    if self.config.target.startswith('avec2014.img.bicnn.normal'):
      net = tf.concat([net0, net1], axis=1)
    elif self.config.target.startswith('avec2014.img.bicnn.rgb'):
      net = tf.concat([net0, net1, logit1], axis=1)
    elif self.config.target.startswith('avec2014.img.bicnn.opt'):
      net = tf.concat([net0, net1, logit0], axis=1)
    else:
      raise ValueError('Unknown input target [%s]' % self.config.target)

    logit = ops.linear(net, 1, 'logits')
    return logit, net

  def _loss(self, logit, label):
    loss = l2.loss(logit, label, self.config)
    return loss

  def _error(self, logit, label):
    mae, rmse = l2.error(logit, label, self.config)
    return mae, rmse

  def train(self):
    self._enter_('train')
    with tf.Graph().as_default() as graph:
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
          func_test=self.test,
          func_val=None))

      saver = tf.train.Saver(var_list=variable.all())
      with context.DefaultSession(self.hooks) as sess:
        self.snapshot.restore(sess, saver)
        while not sess.should_stop():
          sess.run(train_op)

    self._exit_()

  def test(self):
    self._enter_('test')
    # create a folder to save
    test_dir = filesystem.mkdir(self.config.output_dir + '/test/')
    # get data
    image, label, path = load_data(self.config)
    path = tf.unstack(path, axis=1)
    path = path[0]
    # get net
    logit, _ = self._net(image)
    # output to file
    info = string.concat(self.batchsize, [path, label, logit*self.data.span])
    saver = tf.train.Saver()

    # running test
    with context.DefaultSession() as sess:
      global_step = self.snapshot.restore(sess, saver)
      result_path = test_dir + '%s.txt' % global_step
      with open(test_dir + '%s.txt' % global_step, 'wb') as fw:
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

      self._exit_()
      return _rmse