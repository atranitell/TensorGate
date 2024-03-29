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
"""2018/09/27 AVEC2014 """

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

import math
import matplotlib.pyplot as plt


class AVEC2014_IMG_CNN_EX1(context.Context):
  """
  X->(Extractor)->Xf
  1) Xf->Wi->Xi->(Classifier)->Pc->FaceID
  2) Xf->Wd->Xd
    a. Xd->(Regressor)->Pd->ScoreDepression
    b. Xd->(Generator)->Xg->(Regressor)->Pg->ScoreDepression

  Loss:
  1.a) Lc = softmax(Pc-Yc)
    b) Lo: Loss_Orthgonality(Wi, Wd)
  2.a) L_Wd_R = ||Pd-Yr||
    b) L_Wd_G = ||Xd-Xg||
    c) L_G_R = -||Pg-Yr||

  ps: Yr: the regression of groundtruth
      Yc: the classification of groundtruth

  Train(undo):
  # 1) Lc, Lo: resnet_50, C, Wi
  # 2) L_Wd_R, Lo: resnet_50, R, Wd
  # 3) L_G_R, L_Wd_G: G, R
  """

  def __init__(self, config):
    self.num_identity = 98
    self.span = 64.0
    self.feature_dim = 2048
    context.Context.__init__(self, config)

  def _net(self, data):
    _, nets = net_graph(data, self.config.net[0], self.phase)
    Xf = tf.squeeze(
        nets['global_pool'],
        name=nets['global_pool'].name.split('/')[0]+'/last_squeeze')
    # space transpose
    Xi, Wi = self._Wi(Xf)
    Xd, Wd = self._Wd(Xf)
    # print
    # logger.net('Xf: %s' % str(Xf))
    # logger.net('Xi: %s' % str(Xi))
    # logger.net('Wi: %s' % str(Wi))
    # logger.net('Xd: %s' % str(Xd))
    # logger.net('Wd: %s' % str(Wd))
    return Xi, Xd, Wi, Wd, Xf

  def _Wi(self, Xf):
    with tf.variable_scope('Wi'):
      W = tf.Variable(tf.random_normal([2048, self.feature_dim], stddev=0.01))
      X = tf.matmul(Xf, W)
    return X, W

  def _Wd(self, Xf):
    with tf.variable_scope('Wd'):
      W = tf.Variable(tf.random_normal([2048, self.feature_dim], stddev=0.01))
      X = tf.matmul(Xf, W)
    return X, W

  def _classifer(self, X, reuse=None):
    with tf.variable_scope('classifier', reuse=reuse):
      net = tf.expand_dims(X, axis=1)
      net = tf.expand_dims(net, axis=2)
      net = tf.layers.conv2d(
          net, self.num_identity, [1, 1], name='cls',
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
      net = tf.squeeze(net)
      # net = tf.layers.dense(
      #     inputs=X, units=self.num_identity,
      #     kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    return net

  def _generator(self, X, reuse=None):
    with tf.variable_scope('generator', reuse=reuse):
      # net = tf.layers.dense(
      #     inputs=X, units=512, activation=tf.nn.relu,
      #     kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
      # net = tf.layers.dense(
      #     inputs=net, units=1024, activation=tf.nn.relu,
      #     kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
      net = tf.layers.dense(
          inputs=X, units=self.feature_dim,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    return net

  def _regressor(self, X, reuse=None):
    with tf.variable_scope('regressor', reuse=reuse):
      net = tf.expand_dims(X, axis=1)
      net = tf.expand_dims(net, axis=2)
      net = tf.layers.conv2d(
          net, 1, [1, 1], name='reg',
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
      net = tf.squeeze(net)
      # net = tf.layers.dense(
      #     inputs=X, units=1,
      #     kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    return net

  def _loss(self, logit, label):
    """ logit softmax(self.batchsize, self.data.num_classes)
    """

  @context.graph_phase_wrapper()
  def train(self):
    # load data
    image, label, path = load_data(self.config)
    Yc, Yr = tf.unstack(label, axis=1)
    Yr = tf.to_float(Yr) / self.span
    # load net
    Xi, Xd, Wi, Wd, Xf = self._net(image)

    # part1
    Pc = self._classifer(Xi)
    # 1.b) orthgonality loss
    Lorth = tf.reduce_sum(tf.abs(tf.matmul(Wi, Wd, transpose_a=True)))
    # 1.a) classification loss
    Lc = tf.losses.sparse_softmax_cross_entropy(labels=Yc, logits=Pc)

    # part2
    Pd = self._regressor(Xd)
    Xg = self._generator(Xd)
    Pg = self._regressor(Xg, reuse=True)

    # 2.a) Pd, Yr - l2 grounth truth
    Ldr = tf.nn.l2_loss([Yr-Pd], name='L_dr')
    # 2.b)
    Ldg = tf.nn.l2_loss([Xg-Xd], name='L_dg')
    # 2.c)
    Lgr = -tf.nn.l2_loss([tf.to_float(Yr)-Pg], name='L_dg')

    # total loss
    loss = Lorth + Lc + Ldr + Ldg + Lgr

    var_b = variable.select_vars('resnet_v2_50')
    var_wi = variable.select_vars('Wi')
    var_wd = variable.select_vars('Wd')
    var_c = variable.select_vars('classifier')
    var_r = variable.select_vars('regressor')
    var_g = variable.select_vars('generator')

    """Optimize sequence
    s1. (Loss_Wd_R, Loss_orth)->Optimize(resnet_50, Wd, Regressor)
    s2. (Loss_C, Loss_orth)->Optimize(Classifier, Wi)
    s3. (Loss_G, Loss_GR)->Optimize(Generator, Regressor)
    """

    L_0 = Ldr + Lorth
    var_0 = var_b + var_wd + var_r

    L_1 = Lc + Lorth
    var_1 = var_c + var_wi

    L_2 = Ldg + Lgr
    var_2 = var_g + var_r

    # L_2 = Lorth
    # var_2 = var_wd + var_wi

    # L_3 = Ldr + Lorth
    # var_3 = var_b + var_wd + var_r

    # error
    error = (Pd-Yr)*self.span
    mae = tf.reduce_mean(tf.abs(error, name='error_mae'))
    rmse = tf.sqrt(tf.reduce_mean(tf.square(error, name='error_mse')))

    # update gradients
    global_step = tf.train.create_global_step()
    train_op0 = updater.default(self.config, L_0, global_step, var_0)
    train_op1 = updater.default(self.config, L_1, None, var_1)
    train_op2 = updater.default(self.config, L_2, None, var_2, 1)
    # train_op3 = updater.default(self.config, L_3, None, var_3)
    # train_op4 = updater.default(self.config, L_4, None, var_4, 1)
    train_op = tf.group(train_op0, train_op1, train_op2)

    # add hooks
    self.add_hook(self.snapshot.init())
    self.add_hook(self.summary.init())
    self.add_hook(context.Running_Hook(
        config=self.config.log,
        step=global_step,
        keys=['loss', 'mae', 'rmse', 'l_c', 'l_orth', 'l_dr', 'l_dg', 'l_gr'],
        values=[loss, mae, rmse, Lc, Lorth, Ldr, Ldg, Lgr],
        func_test=self.test,
        func_val=None))

    saver = tf.train.Saver(var_list=variable.all())
    with context.DefaultSession(self.hooks) as sess:
      self.snapshot.restore(sess, saver)
      while not sess.should_stop():
        sess.run(train_op)
        # sess.run(train_op1)
        # sess.run(train_op2)
        # sess.run(train_op3)
        # sess.run(train_op4)

  @context.graph_phase_wrapper()
  def test(self):
    # create a folder to save
    test_dir = filesystem.mkdir(self.config.output_dir + '/test/')
    # get data
    image, label, path = load_data(self.config)
    # get net
    _, Xd, _, _, _ = self._net(image)
    pred = self._regressor(Xd)
    # output to file
    info = string.concat(self.batchsize, [path, label, pred*self.span])
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
