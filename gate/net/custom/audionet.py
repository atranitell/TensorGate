
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
"""AudioNet"""

import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers
import tensorflow.contrib.rnn as rnn


def cat(str_v, int_v):
  return str_v + '_' + str(int_v)


def itos(int_list):
  _str = ''
  for i in int_list:
    _str += str(i)
  return _str


class AudioNet():

  def arguments_scope(self):
    with arg_scope([]) as sc:
      return sc

  def model(self, inputs, num_classes, is_training):
    self.is_training = is_training
    self.activation_fn = tf.nn.leaky_relu
    self.weight_decay = 0.0001
    self.batch_norm_decay = 0.999
    self.batch_norm_epsilon = 1e-5
    self.batch_norm_scale = True
    self.dropout = 0.5
    return self.audionet(inputs, num_classes, is_training, 'sen',  17, 'ma')

  def audionet(self, inputs, num_classes, is_training, net_kind, layer_num, keep_type):
    """ a factory corresponding to different experiments"""

    keep = self.choose_keep_type(keep_type)
    layer = self.choose_layer_num(layer_num)

    if net_kind == 'sen':
      return self.model_sen(inputs, num_classes, is_training, keep, layer, keep_type)
    elif net_kind == 'sen_rnn':
      return self.model_sen_rnn(inputs, num_classes, is_training, keep, layer, keep_type)
    else:
      raise ValueError('Unknown network.')

  def choose_keep_type(self, keep_type):
    if keep_type == 'plain':
      return self.keep_plain
    elif keep_type == 'ma':
      return self.keep_ma
    elif keep_type == 'sa':
      return self.keep_sa
    elif keep_type == 'mc':
      return self.keep_mc
    elif keep_type == 'sc':
      return self.keep_sc
    elif keep_type == 'sa_se':
      return self.keep_sa_se
    elif keep_type == 'ia':
      return self.keep_inception_add
    elif keep_type == 'next':
      return self.keep_next

  def choose_layer_num(self, layer_num):
    if layer_num == 9:
      return [1, 1, 0, 0]
    elif layer_num == 17:
      return [1, 1, 1, 1]
    elif layer_num == 29:
      return [1, 3, 3, 1]
    elif layer_num == 35:
      return [2, 4, 4, 2]
    elif layer_num == 53:
      return [3, 5, 5, 3]
    elif layer_num == 71:
      return [3, 5, 1, 3]
    elif layer_num == 113:
      return [3, 13, 17, 3]
    elif layer_num == 161:
      return [6, 17, 23, 6]

  def pool1d(self, inputs, kernel_size, strides,
             pooling_type='MAX', padding_type='SAME', name=None):
    return tf.nn.pool(inputs, [kernel_size], pooling_type,
                      padding_type, strides=[strides], name=name)

  def bn(self, inputs):
    # return inputs
    return layers.batch_norm(
        inputs, decay=self.batch_norm_decay,
        updates_collections=None,
        is_training=self.is_training,
        epsilon=self.batch_norm_epsilon,
        scale=self.batch_norm_scale)

  def conv(self, x, num_filters, k_size, k_stride, name, act=True):
    with tf.variable_scope(name):
      if act is True:
        activation_fn = self.activation_fn
      else:
        activation_fn = None
      net = tf.layers.conv1d(
          inputs=x,
          filters=num_filters,
          kernel_size=k_size,
          strides=k_stride,
          padding='SAME',
          activation=activation_fn,
          kernel_initializer=tf.truncated_normal_initializer(
              stddev=0.01),
          kernel_regularizer=layers.l2_regularizer(self.weight_decay),
          bias_initializer=tf.constant_initializer(0.0))
      return net

  def conv_bn(self, x, num_filters, k_size, k_stride, name, act=True, bn=True):
    with tf.variable_scope(name):
      net = tf.layers.conv1d(
          inputs=x,
          filters=num_filters,
          kernel_size=k_size,
          strides=k_stride,
          padding='SAME',
          activation=None,
          kernel_initializer=tf.truncated_normal_initializer(
              stddev=0.01),
          kernel_regularizer=layers.l2_regularizer(self.weight_decay),
          bias_initializer=tf.constant_initializer(0.0))
      if bn is True:
        net = self.bn(net)
      if act is True:
        net = self.activation_fn(net)
      return net

  def conv_pre_bn(self, x, num_filters, k_size, k_stride, name, act=True):
    if act is True:
      x = self.activation_fn(self.bn(x))
    with tf.variable_scope(name):
      net = tf.layers.conv1d(
          inputs=x,
          filters=num_filters,
          kernel_size=k_size,
          strides=k_stride,
          padding='SAME',
          activation=None,
          kernel_initializer=tf.truncated_normal_initializer(
              stddev=0.01),
          kernel_regularizer=layers.l2_regularizer(self.weight_decay),
          bias_initializer=tf.constant_initializer(0.0))
      return net

  def activation_out(self, x, act_out):
    if act_out:
      return self.activation_fn(x)
    else:
      return x

  def fc(self, x, channels, name='fc'):
    return layers.fully_connected(
        x, channels,
        biases_initializer=None,
        weights_initializer=tf.truncated_normal_initializer(
            stddev=0.01),
        weights_regularizer=layers.l2_regularizer(self.weight_decay),
        activation_fn=None,
        scope=name)

  def keep_inception_add(self, x, in_filters, k_size, name, act_out=True):
    """ x(n) - 1x1(2n) - 3x3(n/2) - 3x3(n) - b1
        x(n) - 1x1(2n) - 3x3(n) - b2
        x(n) - 1x1(n) - b3
        out(n) = x + b1 + b2 + b3
    """
    with tf.variable_scope('b1_' + name):
      b1 = self.conv(x, in_filters * 2, k_size, 1, 'conv1_1')
      b1 = self.conv(b1, int(in_filters / 2), k_size * 3, 1, 'conv1_2')
      b1 = self.conv(b1, in_filters, k_size * 3, 1, 'conv1_3')
    with tf.variable_scope('b2_' + name):
      b2 = self.conv(x, in_filters * 2, k_size, 1, 'conv2_1')
      b2 = self.conv(b2, in_filters, k_size * 3, 1, 'conv2_2')
    with tf.variable_scope('b3_' + name):
      b3 = self.conv(x, in_filters, k_size, 1, 'conv3_1')
    with tf.variable_scope('ia_' + name):
      return self.activation_out(x + b1 + b2 + b3, act_out)

  def keep_sa(self, x, in_filters, k_size, name, act_out=True):
    """ single addition """
    with tf.variable_scope('sa_' + name):
      net = self.conv(x, in_filters, k_size, 1, 'conv1_1')
      net = self.conv(net, in_filters, k_size * 3, 1, 'conv1_2')
      net = self.conv(net, in_filters, k_size, 1, 'conv1_3')
      return self.activation_out(x + net, act_out)

  def keep_next(self, x, in_filters, k_size, name, act_out=True):
    """ single addition """
    with tf.variable_scope('next_' + name):
      cardinality = []
      for i in range(32):
        net = self.conv(x, int(in_filters / 64),
                        k_size, 1, 'conv' + str(i) + '_1')
        net = self.conv(net, int(in_filters / 64),
                        k_size * 3, 1, 'conv' + str(i) + '_2')
        net = self.conv(net, in_filters, k_size,
                        1, 'conv' + str(i) + '_3')
        cardinality.append(net)
      net = cardinality[0]
      for i in range(1, 32):
        net += cardinality[i]
      return self.activation_out(net, act_out)

  def keep_sa_se(self, x, in_filters, k_size, name, act_out=True):
    """ single addition + squeeze"""
    with tf.variable_scope('sa_' + name):
      net = self.conv(x, in_filters * 2, k_size, 1, 'conv1_1')
      net = self.conv(net, in_filters, k_size * 3, 1, 'conv1_2')
      net_res = self.conv(net, in_filters, k_size, 1, 'conv1_3')
    with tf.variable_scope('se_' + name):
      net = tf.reduce_sum(net_res, axis=1)
      net = self.fc(net, int(in_filters / 16), 'fc_1')
      net = self.fc(net, in_filters, 'fc_2')
      scale = tf.sigmoid(net, 'sigmoid')
      scale = tf.reshape(scale, [-1, 1, in_filters])
    with tf.variable_scope('sa_se_' + name):
      return self.activation_out(x + net_res * scale, act_out)

  def sequeeze_out(self, x, in_filters, name):
    with tf.variable_scope('seq_' + name):
      net = tf.reduce_sum(x, axis=1)
      net = self.fc(net, int(in_filters / 16), 'fc_1')
      net = self.fc(net, in_filters, 'fc_2')
      scale = tf.sigmoid(net, 'sigmoid')
      scale = tf.reshape(scale, [-1, 1, in_filters])
      return x * scale

  def keep_sc(self, x, in_filters, k_size, name, act_out=True):
    """ single concat """
    with tf.variable_scope('sc_' + name):
      x1 = self.conv(x, in_filters, k_size, 1, 'conv1_1')
      x2 = self.conv(x1, in_filters, k_size * 2, 1, 'conv1_2')
      x3_c = tf.concat([x, x2], axis=2)
      x3 = self.conv(x3_c, in_filters, k_size, 1, 'conv1_3')
      return self.activation_out(x3, act_out)

  def keep_mc(self, x, in_filters, k_size, name, act_out=True):
    """ multi concat """
    with tf.variable_scope('mc_' + name):
      x1 = self.conv(x, in_filters, k_size, 1, 'conv1_1')
      x1_c = tf.concat([x, x1], axis=2)
      x2 = self.conv(x1_c, in_filters, k_size * 2, 1, 'conv1_2')
      x2_c = tf.concat([x, x1, x2], axis=2)
      x3 = self.conv(x2_c, in_filters, k_size, 1, 'conv1_3')
      return self.activation_out(x + x3, act_out)

  def keep_ma(self, x, in_filters, k_size, name, act_out=True):
    """ multi addition """
    with tf.variable_scope('ma_' + name):
      x1 = self.conv(x, in_filters, k_size, 1, 'conv1_1')
      x2 = self.conv(x + x1, in_filters, k_size * 2, 1, 'conv1_2')
      x3 = self.conv(x + x2, in_filters, k_size, 1, 'conv1_3')
      return self.activation_out(x + x3, act_out)

  def keep_plain(self, x, in_filters, k_size, name, act_out=True):
    """ conv1 -> conv2 -> conv3 """
    with tf.variable_scope('plain_' + name):
      x1 = self.conv(x, in_filters, k_size, 1, 'conv1_1')
      x2 = self.conv(x1, in_filters, k_size * 2, 1, 'conv1_2')
      x3 = self.conv(x2, in_filters, k_size, 1, 'conv1_3')
      return x3

  def keep_ma_bn(self, x, in_filters, k_size, name, act_out=True):
    """ conv1 -> conv2 -> conv3 """
    with tf.variable_scope('ma_bn_' + name):
      x1 = self.conv_bn(x, in_filters, k_size, 1, 'conv1_1')
      x2 = self.conv_bn(x + x1, in_filters, k_size * 2, 1, 'conv1_2')
      x3 = self.conv_bn(x + x2, in_filters, k_size, 1, 'conv1_3', False)
      return self.activation_out(x + x3, act_out)

  def keep_ma_pre_bn(self, x, in_filters, k_size, name, act_out=True):
    """ conv1 -> conv2 -> conv3 """
    with tf.variable_scope('ma_prebn_' + name):
      x1 = self.conv_pre_bn(x, in_filters, k_size, 1, 'conv1_1')
      x2 = self.conv_pre_bn(x + x1, in_filters, k_size * 2, 1, 'conv1_2')
      x3 = self.conv_pre_bn(x + x2, in_filters, k_size, 1, 'conv1_3')
      return x + x3

  def attention_module(self, x, feature_map, name):
    """
    """
    with tf.variable_scope('am_' + name):
      fn, fw, fc = feature_map.get_shape().as_list()
      dn, dw, dc = x.get_shape().as_list()
      assert fn == dn
      scale = int(dw / fw)
      attention = tf.sigmoid(self.pool1d(x, scale, scale, pooling_type='AVG'))
      return feature_map * attention

  def model_sen2(self, inputs, num_classes, is_training, keep, num_block, keep_type):
    """
    """
    end_points = {}

    with tf.variable_scope('sen1_' + keep_type + '_' + itos(num_block)):
      # block1-6400
      net = self.conv(inputs, 64, 20, 2, name='conv0')
      net = self.pool1d(net, 2, 2, name='pool0')
      # block1-1600
      for i in range(num_block[0]):
        net = keep(net, 64, 1, cat('k1', i))
      net_k1 = self.attention_module(inputs, net, 'a1')

      # block2-1600
      net = self.conv(net_k1, 128, 10, 2, 'k1')
      net = self.pool1d(net, 2, 2, name='pool1')
      # block2-400
      for i in range(num_block[1]):
        net = keep(net, 128, 1, cat('k2', i))
      net_k2 = self.attention_module(inputs, net, 'a2')

      # block3-400
      net = self.conv(net_k2, 128, 10, 2, 'k2')
      net = self.pool1d(net, 2, 2, name='pool2')
      # block3-100
      for i in range(num_block[2]):
        net = keep(net, 128, 1, cat('k3', i))
      net_k3 = self.attention_module(inputs, net, 'a3')

      # block4-100
      net = self.conv(net_k3, 256, 10, 2, 'k3')
      net = self.pool1d(net, 2, 2, name='pool3')
      # block4-25
      for i in range(num_block[3] - 1):
        net = keep(net, 256, 1, cat('k4', i))
      net_k4 = keep(net, 256, 1, cat('k4', num_block[3]), 'False')

      # scale to same length
      net_k1 = self.pool1d(net_k1, 64, 64, 'AVG', name='k1_pool')
      net_k2 = self.pool1d(net_k2, 16, 16, 'AVG', name='k2_pool')
      net_k3 = self.pool1d(net_k3, 4, 4, 'AVG', name='k3_pool')

      # seqeeze
      net_k1 = self.sequeeze_out(net_k1, 64, 'k1')
      net_k2 = self.sequeeze_out(net_k2, 128, 'k2')
      net_k3 = self.sequeeze_out(net_k3, 128, 'k3')

      # concat
      net = tf.concat([net_k4, net_k3, net_k2, net_k1], axis=2)

      end_points['gap_conv'] = net
      end_points['net_k1'] = net_k1
      end_points['net_k2'] = net_k2
      end_points['net_k3'] = net_k3
      end_points['net_k4'] = net_k4

      net = tf.reduce_sum(net, axis=1)
      net = layers.dropout(net, self.dropout, is_training=is_training)

      logits = layers.fully_connected(
          net, num_classes,
          biases_initializer=None,
          weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
          weights_regularizer=None,
          activation_fn=None,
          scope='logits')

      return logits, end_points

  def model_sen(self, inputs, num_classes, is_training, keep, num_block, keep_type):
    """
    """
    end_points = {}

    with tf.variable_scope('sen1_' + keep_type + '_' + itos(num_block)):
      # block1-6400
      net = self.conv(inputs, 64, 20, 2, name='conv0')
      net = self.pool1d(net, 2, 2, name='pool0')
      # block1-1600
      for i in range(num_block[0]):
        net = keep(net, 64, 1, cat('k1', i))
      net_k1 = net
      end_points['k1'] = net_k1
      net_k1 = self.attention_module(inputs, net_k1, 'a1')

      # block2-1600
      net = self.conv(net, 128, 10, 2, 'k1')
      net = self.pool1d(net, 2, 2, name='pool1')
      # block2-400
      for i in range(num_block[1]):
        net = keep(net, 128, 1, cat('k2', i))
      net_k2 = net
      end_points['k2'] = net_k2
      net_k2 = self.attention_module(inputs, net_k2, 'a2')

      # block3-400
      net = self.conv(net, 128, 10, 2, 'k2')
      net = self.pool1d(net, 2, 2, name='pool2')
      # block3-100
      for i in range(num_block[2]):
        net = keep(net, 128, 1, cat('k3', i))
      net_k3 = net
      end_points['k3'] = net_k3
      net_k3 = self.attention_module(inputs, net_k3, 'a3')

      # block4-100
      net = self.conv(net, 256, 10, 2, 'k3')
      net = self.pool1d(net, 2, 2, name='pool3')
      # block4-25
      for i in range(num_block[3] - 1):
        net = keep(net, 256, 1, cat('k4', i))
      net = keep(net, 256, 1, cat('k4', num_block[3]), 'False')
      net_k4 = net
      end_points['k4'] = net_k4

      # scale to same length
      net_k1 = self.pool1d(net_k1, 64, 64, 'AVG', name='k1_pool')
      net_k2 = self.pool1d(net_k2, 16, 16, 'AVG', name='k2_pool')
      net_k3 = self.pool1d(net_k3, 4, 4, 'AVG', name='k3_pool')

      # seqeeze
      net_k1 = self.sequeeze_out(net_k1, 64, 'k1')
      net_k2 = self.sequeeze_out(net_k2, 128, 'k2')
      net_k3 = self.sequeeze_out(net_k3, 128, 'k3')

      # concat
      net = tf.concat([net_k4, net_k3, net_k2, net_k1], axis=2)
      end_points['gap_conv'] = net

      net = tf.reduce_sum(net, axis=1)
      net = layers.dropout(net, self.dropout, is_training=is_training)

      logits = layers.fully_connected(
          net, num_classes,
          biases_initializer=None,
          weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
          weights_regularizer=None,
          activation_fn=None,
          scope='logits')

      return logits, end_points

  def model_sen1(self, inputs, num_classes, is_training, keep, num_block, keep_type):
    """
    """
    end_points = {}

    with tf.variable_scope('sen1_' + keep_type + '_' + itos(num_block)):
      # root-12800
      net = self.conv(inputs, 64, 20, 2, name='conv0')
      net = self.pool1d(net, 2, 2, name='pool0')

      # block1-3200
      for i in range(num_block[0]):
        net = keep(net, 64, 1, cat('k1', i))
      net = self.conv(net, 128, 10, 2, 'k1')
      net = self.pool1d(net, 2, 2, name='pool1')

      # block2-800
      for i in range(num_block[1]):
        net = keep(net, 128, 1, cat('k2', i))
      net_k2 = net
      net = self.conv(net, 128, 10, 2, 'k2')
      net = self.pool1d(net, 2, 2, name='pool2')

      # block3-200
      for i in range(num_block[2]):
        net = keep(net, 128, 1, cat('k3', i))
      net_k3 = net
      net = self.conv(net, 256, 10, 2, 'k3')
      net = self.pool1d(net, 2, 2, name='pool3')

      # block3-50
      for i in range(num_block[3] - 1):
        net = keep(net, 256, 1, cat('k4', i))
      net_k4 = keep(net, 256, 1, cat('k4', num_block[3]), False)

      # scale to same length
      net_k2 = self.pool1d(net_k2, 16, 16, 'AVG', name='k2_pool')
      net_k3 = self.pool1d(net_k3, 4, 4, 'AVG', name='k3_pool')

      # seqeeze
      net_k2 = self.sequeeze_out(net_k2, 128, 'k2')
      net_k3 = self.sequeeze_out(net_k3, 128, 'k3')
      # net_k4 = self.sequeeze_out(net_k4, 256, 'k4')

      # concat
      net = tf.concat([net_k4, net_k3, net_k2], axis=2)

      end_points['net_k4'] = net
      end_points['net_k3'] = net
      end_points['net_k2'] = net

      end_points['gap_conv'] = net
      net = tf.reduce_sum(net, axis=1)
      net = layers.dropout(net, self.dropout, is_training=is_training)

      logits = layers.fully_connected(
          net, num_classes,
          biases_initializer=None,
          weights_initializer=tf.truncated_normal_initializer(
              stddev=0.01),
          weights_regularizer=None,
          activation_fn=None,
          scope='logits')

      return logits, end_points

  def model_sen_rnn(self, inputs, num_classes, is_training,
                    keep, num_block, keep_type):
    """
    """
    end_points = {}

    with tf.variable_scope('sen_rnn_' + keep_type + '_' + itos(num_block)):
      # root-12800
      net = self.conv(inputs, 64, 20, 2, name='conv0')
      net = self.pool1d(net, 2, 2, name='pool0')

      # block1-3200
      for i in range(num_block[0]):
        net = keep(net, 64, 1, cat('k1', i))
      net = self.conv(net, 128, 10, 2, 'k1')
      net = self.pool1d(net, 2, 2, name='pool1')

      # block2-800
      for i in range(num_block[1]):
        net = keep(net, 128, 1, cat('k2', i))
      net_k2 = net
      net = self.conv(net, 128, 10, 2, 'k2')
      net = self.pool1d(net, 2, 2, name='pool2')

      # block3-200
      for i in range(num_block[2]):
        net = keep(net, 128, 1, cat('k3', i))
      net_k3 = net
      net = self.conv(net, 256, 10, 2, 'k3')
      net = self.pool1d(net, 2, 2, name='pool3')

      # block3-50
      for i in range(num_block[3] - 1):
        net = keep(net, 256, 1, cat('k4', i))
      net_k4 = keep(net, 256, 1, cat('k4', num_block[3]), False)

      # scale to same length
      net_k2 = self.pool1d(net_k2, 16, 16, 'AVG', name='k2_pool')
      net_k3 = self.pool1d(net_k3, 4, 4, 'AVG', name='k3_pool')

      # seqeeze
      net_k2 = self.sequeeze_out(net_k2, 128, 'k2')
      net_k3 = self.sequeeze_out(net_k3, 128, 'k3')
      # net_k4 = self.sequeeze_out(net_k4, 256, 'k4')

      # concat
      net = tf.concat([net_k4, net_k3, net_k2], axis=2)

      # (N, 50, 512)
      net = self.sens_rnn(net)

      end_points['gap_conv'] = net
      net = tf.reduce_sum(net, axis=1)
      net = layers.dropout(net, self.dropout, is_training=is_training)

      logits = layers.fully_connected(
          net, num_classes,
          biases_initializer=None,
          weights_initializer=tf.truncated_normal_initializer(
              stddev=0.01),
          weights_regularizer=None,
          activation_fn=None,
          scope='logits')

      return logits, end_points

  def sens_rnn(self, X):
    """ input args
    a. num_layers
    b. timesteps
    c. cell_fn
    d. activation_fn
    e. batch_size
    f. num_units
    """
    activation_fn = component.activation_fn('relu')
    cell_fn = component.rnn_cell('gru')
    initializer_fn = component.initializer_fn('orthogonal')

    num_units = 128
    num_layers = 1

    # X shape is [batchsize, time_step, feature]
    n_steps = 50
    n_dim = 512
    batch_size = X.get_shape().as_list()[0]

    # reshape
    # print(X)
    # X = tf.reshape(X, [-1, n_steps, n_dim])

    # transform to list
    X = tf.unstack(X, n_steps, axis=1)

    # sequence_length
    sequence_length = [n_dim for _ in range(batch_size)]

    # multi-layers
    hidden_input = X
    for idx_layer in range(num_layers):
      scope = 'layer_' + str(idx_layer + 1)

      # define
      forward_cell = cell_fn(
          num_units, activation=activation_fn)
      backward_cell = cell_fn(
          num_units, activation=activation_fn)

      # brnn
      # forward-backward, forward final state, backward final state
      fbH, fst, bst = rnn.static_bidirectional_rnn(
          forward_cell, backward_cell, hidden_input, dtype=tf.float32,
          sequence_length=sequence_length, scope=scope)

      fbHrs = [tf.reshape(t, [batch_size, 2, num_units])
               for t in fbH]

      if idx_layer != num_layers - 1:
        # output size is [seqlength, batchsize, 2, num_units]
        output = tf.convert_to_tensor(fbHrs, dtype=tf.float32)

        # output size is [seqlength, batchsize, num_units]
        output = tf.reduce_sum(output, 2)

        # from [seqlenth, batchsize, num_units]
        # to [batchsize, seqlenth, num_units]
        hidden_input = tf.unstack(
            tf.transpose(output, [1, 0, 2]), n_steps, axis=1)

    # sum fw and bw
    # [num_steps, batchsize, n_dim]
    output = tf.convert_to_tensor(fbHrs, dtype=tf.float32)
    output = tf.reduce_sum(output, axis=2)
    output = tf.transpose(output, [1, 0, 2])
    return output
