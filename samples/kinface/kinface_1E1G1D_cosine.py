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


class KINFACE_1E1G1D_COSINE(kinbase.KINBASE):

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
    c1_mu, c1_sigma, feat_c1 = self._encoder(c1_real)
    p2_mu, p2_sigma, feat_p2 = self._encoder(p2_real, True)

    C_loss, _ = self._loss_metric(feat_c1, feat_p2, label)

    # children to parent
    c1_z = c1_mu + c1_sigma * tf.random_normal(tf.shape(c1_mu))
    c1_z = self._generator(c1_z, cond)
    c1_fake = tf.clip_by_value(c1_z, 1e-8, 1 - 1e-8)

    # parent to children
    p2_z = p2_mu + p2_sigma * tf.random_normal(tf.shape(p2_mu))
    p2_z = self._generator(p2_z, cond, True)
    p2_fake = tf.clip_by_value(p2_z, 1e-8, 1 - 1e-8)

    # discriminator
    D_c1_fake = self._discriminator(c1_fake, cond)
    D_p1_real = self._discriminator(p1_real, cond, reuse=True)
    D_p2_fake = self._discriminator(p2_fake, cond, reuse=True)
    D_c2_real = self._discriminator(c2_real, cond, reuse=True)

    # loss for genertor
    E1_loss = self._loss_vae(p1_real, c1_fake, c1_mu, c1_sigma)
    E2_loss = self._loss_vae(c2_real, p2_fake, p2_mu, p2_sigma)
    E_loss = E1_loss + E2_loss

    # loss for discriminator
    D1_loss, G1_loss = self._loss_gan(D_c1_fake, D_p1_real)
    D2_loss, G2_loss = self._loss_gan(D_p2_fake, D_c2_real)
    D_loss = D1_loss + D2_loss
    G_loss = G1_loss + G2_loss

    loss = E_loss + D_loss + G_loss + C_loss

    # update gradients
    global_step = tf.train.create_global_step()

    var_e = variable.select_vars('encoder')
    var_g = variable.select_vars('generator')
    var_d = variable.select_vars('discriminator')

    op1 = updater.default(self.config, loss, global_step, var_e, 0)
    op2 = updater.default(self.config, loss, None, var_g, 1)
    op3 = updater.default(self.config, loss, None, var_d, 0)
    train_op = tf.group(op1, op2, op3)

    # add hooks
    self.add_hook(self.snapshot.init())
    self.add_hook(self.summary.init())
    self.add_hook(context.Running_Hook(
        config=self.config.log,
        step=global_step,
        keys=['E', 'D', 'G', 'C'],
        values=[E_loss, D_loss, G_loss, C_loss],
        func_test=self.test,
        func_val=None))

    saver = tf.train.Saver(var_list=variable.all())
    with context.DefaultSession(self.hooks) as sess:
      self.snapshot.restore(sess, saver)
      while not sess.should_stop():
        sess.run(train_op)
