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
"""FOR KINFACE"""

import tensorflow as tf
from gate import context
from gate.data.data_factory import load_data
from gate.solver import updater
from gate.utils import variable
from samples.kinface import kinbase


class KINFACE_1E(kinbase.KINBASE):

  def __init__(self, config):
    kinbase.KINBASE.__init__(self, config)

  def _net(self, c1_real, p2_real):
    c1_mu, c1_sigma, feat_c1 = self._encoder(c1_real)
    p2_mu, p2_sigma, feat_p2 = self._encoder(p2_real, True)
    return feat_c1, feat_p2

  @context.graph_phase_wrapper()
  def train(self):
    # load data
    data, info, path = load_data(self.config)
    c1_real, p1_real, c2_real, p2_real = tf.unstack(data, axis=1)
    label, cond = tf.unstack(info, axis=1)
    # load net
    feat_c1, feat_p2 = self._net(c1_real, p2_real)
    loss, loss_batch = self._loss_metric(feat_c1, feat_p2, label)
    # update gradients
    global_step = tf.train.create_global_step()
    train_op = updater.default(self.config, loss, global_step)
    # add hooks
    self.add_hook(self.snapshot.init())
    self.add_hook(self.summary.init())
    self.add_hook(context.Running_Hook(
        config=self.config.log,
        step=global_step,
        keys=['R'],
        values=[loss],
        func_test=self.test,
        func_val=None))

    saver = tf.train.Saver(var_list=variable.all())
    with context.DefaultSession(self.hooks) as sess:
      self.snapshot.restore(sess, saver)
      while not sess.should_stop():
        sess.run(train_op)
