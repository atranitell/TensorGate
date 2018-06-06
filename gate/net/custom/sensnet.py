
# Copyright 2017 The KaiJIN Authors. All Rights Reserved.
#
# Licensed under the Apache License, version 2.0 (the "License");
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
"""SensNet.CNN"""

import functools
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers


class SensNet():

  def __init__(self, config, is_training):
    """ SensNet Config:
      name = 'sensnet_v1'
      ---
      activation_fn = tf.nn.leaky_relu / tf.nn.relu
      weight_decay = 0.0001
      batch_norm_decay = 0.999
      batch_norm_epsilon = 1e-5
      batch_norm_scale = True
      use_batch_norm = False
      dropout = 0.5
      ---
      unit_type = 'multi_addition'
      unit_num = [1, 1, 1, 1]
      ---
    """
    self.is_training = is_training
    self.name = config.name

    self.weight_decay = config.weight_decay
    self.activation_fn = config.activation_fn
    self.dropout_keep = config.dropout_keep
    self.num_classes = config.num_classes

    self.use_batch_norm = config.use_batch_norm
    self.use_pre_batch_norm = config.use_pre_batch_norm
    self.batch_norm_decay = config.batch_norm_decay
    self.batch_norm_epsilon = config.batch_norm_decay
    self.batch_norm_scale = config.batch_norm_scale

    self.unit_type = config.unit_type
    self.unit_fn = self.unit_type_selector(self.unit_type)
    if len(config.unit_num) != 4:
      raise ValueError('It should fill in 4 block num (fill 0 if not)')
    self.unit_num = config.unit_num

  def unit_type_selector(self, unit_type):
    if unit_type == 'multi_addition':
      return self.multi_addition_unit
    else:
      raise ValueError('Unknown the unit type [%s]' % unit_type)

  def __call__(self, X, reuse=None):
    """Directly Call"""
    if self.name == 'sensnet_v1':
      return self.SensNet_v1(X, reuse)
    else:
      raise ValueError('Unknown the net [%s]' % self.name)

  @property
  def unit_num_str(self):
    """return 'a_b_c_d' """
    return '_'.join(['%s' % i for i in self.unit_num])

  def linear(self, x, channels, name='fc', use_activation=False):
    """A[M*N] * B[N*P] -> C[M*P]"""
    activation_fn = self.activation_fn if use_activation else None
    return layers.fully_connected(
        x, channels,
        biases_initializer=None,
        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
        weights_regularizer=layers.l2_regularizer(self.weight_decay),
        activation_fn=activation_fn,
        scope=name)

  def pool(self, inputs, kernel_size, strides, name=None,
           pooling_type='MAX', padding_type='SAME'):
    """Default is "MAX" and "SAME"  """
    return tf.nn.pool(inputs, [kernel_size], pooling_type,
                      padding_type, strides=[strides], name=name)

  def conv(self, x, num_filters, k_size, k_stride, name,
           use_activation=True, use_bn=False, use_pre_bn=False):
    """Convolution Layer for 1D CNN
      Normal: Weight + BN + ReLU
      Pre-BN: BN + ReLU + Weight
      No-BN: Weight + ReLU
      None: Weight
    """
    with tf.variable_scope(name):
      net = x
      # pre bn
      if use_pre_bn:
        if use_bn:
          net = self.bn(net)
        if use_activation:
          net = self.activation_fn(net)
      # conv
      net = tf.layers.conv1d(
          inputs=net,
          filters=num_filters,
          kernel_size=k_size,
          strides=k_stride,
          padding='SAME',
          activation=None,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
          kernel_regularizer=layers.l2_regularizer(self.weight_decay),
          bias_initializer=tf.constant_initializer(0.0))
      # after bn
      if not use_pre_bn:
        if use_bn:
          net = self.bn(net)
        if use_activation:
          net = self.activation_fn(net)
      return net

  def bn(self, inputs):
    return layers.batch_norm(
        inputs,
        decay=self.batch_norm_decay,
        updates_collections=None,
        is_training=self.is_training,
        epsilon=self.batch_norm_epsilon,
        scale=self.batch_norm_scale)

  def activation_output(self, x, use_activation_output=False):
    """Use activation to activate inputs"""
    if use_activation_output:
      return self.activation_fn(x)
    else:
      return x

  def multi_addition_unit(self, x, in_filters, k_size, name,
                          use_activation_output=True,
                          use_activation=True,
                          use_bn=False,
                          use_pre_bn=False):
    """Multi-Addition
      A1 = f(X)
      A2 = f(f(A1) + X)
      A3 = f(f(A2) + X)
      Ot = A3 + X
    """
    with tf.variable_scope('ma_' + name):
      x1 = self.conv(x, in_filters, k_size, 1, 'conv1_1',
                     use_activation, use_bn, use_pre_bn)
      x2 = self.conv(x + x1, in_filters, k_size * 2, 1, 'conv1_2',
                     use_activation, use_bn, use_pre_bn)
      x3 = self.conv(x + x2, in_filters, k_size, 1, 'conv1_3',
                     use_activation, use_bn, use_pre_bn)
      return self.activation_output(x + x3, use_activation_output)

  def single_addition_unit(self, x, in_filters, k_size, name,
                           use_activation_output=False,
                           use_activation=True,
                           use_bn=False,
                           use_pre_bn=False):
    """Single Addition
      A1 = f(x)
      A2 = f(A1)
      A3 = f(A2)
      Ot = A3 + x
    """
    with tf.variable_scope('sa_' + name):
      net = self.conv(x, in_filters, k_size, 1, 'conv1_1',
                      use_activation, use_bn, use_pre_bn)
      net = self.conv(net, in_filters, k_size * 3, 1, 'conv1_2',
                      use_activation, use_bn, use_pre_bn)
      net = self.conv(net, in_filters, k_size, 1, 'conv1_3',
                      use_activation, use_bn, use_pre_bn)
      return self.activation_output(x + net, use_activation_output)

  def squeeze_excitation_module(self, x, in_filters, name):
    """Sequeeze and Excitation Module

    See:
      Squeeze-and-Excitation Networks
      http://arxiv.org/abs/1709.01507

    X [N, W, C] -> [N, C] -> [N, filter/16] -> [N, filter]
      -> [N, filter](0, 1) -> [N, 1, filter] -> X[N, W, C] * [N, 1, filter]
    """
    with tf.variable_scope('seq_' + name):
      net = tf.reduce_sum(x, axis=1)
      net = self.linear(net, int(in_filters / 16), 'fc_1', use_activation=True)
      net = self.linear(net, in_filters, 'fc_2', use_activation=False)
      scale = tf.sigmoid(net, 'sigmoid')
      scale = tf.reshape(scale, [-1, 1, in_filters])
      return x * scale

  def attention_module(self, x, feature_map, name):
    """Attention Module"""
    with tf.variable_scope('am_' + name):
      fn, fw, fc = feature_map.get_shape().as_list()
      dn, dw, dc = x.get_shape().as_list()
      assert fn == dn
      scale = int(dw / fw)
      attention = tf.sigmoid(self.pool(x, scale, scale, pooling_type='AVG'))
      return feature_map * attention

  def SensNet_v1(self, X, reuse=None):
    """SensNet including:
      1) residual structure
      2) seqeeze and excitation module
      3) data-based attention re-allocation
      4) multi-view preception
    """
    ends = {}
    model_name = '_'.join(['SensNet_v1', self.unit_type, self.unit_num_str])

    conv = functools.partial(self.conv,
                             use_activation=True,
                             use_bn=self.use_batch_norm,
                             use_pre_bn=self.use_pre_batch_norm)

    pool = functools.partial(self.pool,
                             pooling_type='MAX',
                             padding_type='SAME')

    with tf.variable_scope(model_name):
      # BLOCK1
      net = conv(X, 64, 20, 2, 'conv_1')
      net = pool(net, 2, 2, 'pool_1')
      for i in range(self.unit_num[0]):
        net = self.unit_fn(net, 64, 1, 'unit_1_'+str(i))
      ends['unit_1'] = net

      # BLOCK2
      net = conv(net, 128, 10, 2, 'conv_2')
      net = pool(net, 2, 2, 'pool_2')
      for i in range(self.unit_num[1]):
        net = self.unit_fn(net, 128, 1, 'unit_2_'+str(i))
      ends['unit_2'] = net

      # BLOCK3
      net = conv(net, 128, 10, 2, 'conv_3')
      net = pool(net, 2, 2, 'pool_3')
      for i in range(self.unit_num[2]):
        net = self.unit_fn(net, 128, 1, 'unit_3_'+str(i))
      ends['unit_3'] = net

      # BLOCK4: output without activation
      net = conv(net, 256, 10, 2, 'conv_4')
      net = pool(net, 2, 2, 'pool_4')
      for i in range(self.unit_num[3]-1):
        net = self.unit_fn(net, 256, 1, 'unit_4_'+str(i))
      net = self.unit_fn(net, 256, 1, 'unit_4_'+str(self.unit_num[3]-1))
      ends['unit_4'] = net

      # attention module
      ends['a_1'] = self.attention_module(X, ends['unit_1'], 'a_1')
      ends['a_2'] = self.attention_module(X, ends['unit_2'], 'a_2')
      ends['a_3'] = self.attention_module(X, ends['unit_3'], 'a_3')

      # pool to same length
      # for SE-modlue do not change the length
      ends['p_1'] = self.pool(ends['a_1'], 64, 64, 'p_1', 'AVG')
      ends['p_2'] = self.pool(ends['a_2'], 16, 16, 'p_2', 'AVG')
      ends['p_3'] = self.pool(ends['a_3'], 4, 4, 'p_3', 'AVG')

      # Sequeeze and Excitation module
      ends['se_1'] = self.squeeze_excitation_module(ends['p_1'], 64, 'se_1')
      ends['se_2'] = self.squeeze_excitation_module(ends['p_2'], 128, 'se_2')
      ends['se_3'] = self.squeeze_excitation_module(ends['p_3'], 128, 'se_3')

      # concat
      ends['concat'] = tf.concat([ends['unit_4'], ends['se_3'],
                                  ends['se_2'], ends['se_1']], axis=2)

      # output
      net = tf.reduce_sum(ends['concat'], axis=1)
      net = layers.dropout(net, self.dropout_keep,
                           is_training=self.is_training)
      logits = self.linear(net, self.num_classes, 'logits')

      return logits, ends
