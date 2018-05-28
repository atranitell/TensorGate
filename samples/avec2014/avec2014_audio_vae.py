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
"""2018/5/22 AVEC2014"""

import tensorflow as tf
from gate import context
from gate.net.custom import sens_vae
from gate.data.data_factory import load_data
from gate.solver import updater
from gate.layer import l2
from gate.utils import variable
from gate.utils import filesystem
from gate.utils import string
from gate.utils.logger import logger
from samples.avec2014.utils import get_accurate_from_file


class AVEC2014_AUDIO_VAE(context.Context):

  def __init__(self, config):
    context.Context.__init__(self, config)

  def _encoder(self, data):
    return sens_vae.encoder(data, 1, self.is_training)

  def _generator(self, z):
    return sens_vae.generator(z, self.is_training)

  def _loss_classifier(self, logit, label):
    loss = l2.loss(logit, label, self.config)
    return loss

  def _error(self, logit, label):
    mae, rmse = l2.error(logit, label, self.config)
    return mae, rmse

  def _loss_match(self, real, fake):
    real = tf.expand_dims(real, 3)
    fake = tf.expand_dims(fake, 3)
    marginal_likelihood = 0.000001*tf.reduce_mean(tf.reduce_sum(
        real * tf.log(fake) + (1 - real) * tf.log(1 - fake), [1, 2]))
    return marginal_likelihood

  def _loss_vae(self, fake, real, mu, sigma):
    likelihood = tf.reduce_mean(tf.square(fake-real), [0, 1, 2])
    # marginal_likelihood = tf.reduce_sum(
    #     real * tf.log(fake) + (1 - real) * tf.log(1 - fake), [1, 2])
    # neg_loglikelihood = -tf.reduce_mean(marginal_likelihood)
    KL_divergence = 0.5 * tf.reduce_sum(
        tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, [1])
    KL_divergence = tf.reduce_mean(KL_divergence)
    # ELBO = -neg_loglikelihood - KL_divergence
    return likelihood, KL_divergence

  @context.graph_phase_wrapper()
  def train(self):
    # load data
    data, label, path = load_data(self.config)
    real1, real2 = tf.unstack(data, num=2, axis=1)

    # load net
    c1_mu, c1_sigma, c1_logit = self._encoder(real1)

    # generator
    c1_z = c1_mu + c1_sigma * tf.random_normal(tf.shape(c1_mu))
    c1_z = self._generator(c1_z)
    fake1 = tf.clip_by_value(c1_z, 1e-8, 1 - 1e-8)

    L_loss, KL_loss = self._loss_vae(fake1, real2, c1_mu, c1_sigma)
    C_loss = self._loss_classifier(c1_logit, label)
    loss = L_loss + C_loss + KL_loss

    # compute error
    mae, rmse = self._error(c1_logit, label)

    # update gradients
    global_step = tf.train.create_global_step()
    var_e = variable.select_vars('encoder')
    var_g = variable.select_vars('generator')
    op1 = updater.default(self.config, loss, global_step, var_e, 0)
    op2 = updater.default(self.config, loss, None, var_g, 1)
    train_op = tf.group(op1, op2)

    # add hooks
    self.add_hook(self.snapshot.init())
    self.add_hook(self.summary.init())
    self.add_hook(context.Running_Hook(
        config=self.config.log,
        step=global_step,
        keys=['C_loss', 'L_loss', 'KL_loss', 'mae', 'rmse'],
        values=[C_loss, L_loss, KL_loss, mae, rmse],
        func_test=self.test,
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
    data, label, path = load_data(self.config)

    # load net
    _, _, logit = self._encoder(data)

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
    data, label, path = load_data(self.config)

    # load net
    _, _, logit = self._encoder(data)

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
