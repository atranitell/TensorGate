# -*- coding: utf-8 -*-
""" AudioNet for 1D audio classification or regression.
    Author: Kai JIN
    Updated: 2017/07/21
"""
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers
from gate.net import net


def cat(str_v, int_v):
    return str_v + '_' + str(int_v)


def itos(int_list):
    _str = ''
    for i in int_list:
        _str += str(i)
    return _str


class AudioNet(net.Net):

    def arguments_scope(self):
        with arg_scope([]) as sc:
            return sc

    def model(self, inputs, num_classes, is_training):
        self.is_training = is_training
        self.activation_fn = tf.nn.relu
        return self.audionet(inputs, num_classes, is_training, 'sen', 53, 'ma')

    def audionet(self, inputs, num_classes, is_training, net_kind, layer_num=17, keep_type='ma'):
        """ a factory corresponding to different experiments"""

        keep = self.choose_keep_type(keep_type)
        layer = self.choose_layer_num(layer_num)

        if net_kind == 'sen':
            return self.model_sen(inputs, num_classes, is_training, keep, layer)
        else:
            raise ValueError('Unknown network.')

    def choose_keep_type(self, keep_type):
        if keep_type == 'plain':
            return self.keep_plain
        elif keep_type == 'ma':
            return self.keep_ma
        elif keep_type == 'concat':
            return self.keep_concat
        elif keep_type == 'dense':
            return self.keep_dense
        elif keep_type == 'res':
            return self.keep_res

    def choose_layer_num(self, layer_num):
        if layer_num == 9:
            return [1, 1, 0, 0]
        elif layer_num == 17:
            return [1, 1, 1, 1]
        elif layer_num == 29:
            return [2, 2, 2, 2]
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

    def keep_res(self, x, in_filters, k_size, name, act_out=True):
        """ conv1 -> conv2 -> conv3
        """
        with tf.variable_scope('res_' + name):
            net = self.conv(x, in_filters, k_size, 1, 'conv1_1')
            net = self.conv(net, in_filters, k_size * 2, 1, 'conv1_2')
            net = self.conv(net, in_filters, k_size, 1, 'conv1_3')
            return self.activation_out(x + net, act_out)

    def keep_dense(self, x, in_filters, k_size, name, act_out=True):
        """ conv1 -> conv2 -> conv3
        """
        with tf.variable_scope('dense_' + name):
            x1 = self.conv(x, in_filters, k_size, 1, 'conv1_1')
            x2 = self.conv(x + x1, in_filters, k_size * 2, 1, 'conv1_2')
            x3 = self.conv(x + x1 + x2, in_filters, k_size, 1, 'conv1_3')
            return self.activation_out(x + x1 + x2 + x3, act_out)

    def keep_concat(self, x, in_filters, k_size, name, act_out=True):
        """ conv1 -> conv2 -> conv3
        """
        with tf.variable_scope('concat_' + name):
            x1 = self.conv(x, in_filters, k_size, 1, 'conv1_1')
            x1_c = tf.concat([x, x1], axis=2)
            x2 = self.conv(x1_c, in_filters, k_size * 2, 1, 'conv1_2')
            x2_c = tf.concat([x, x1, x2], axis=2)
            x3 = self.conv(x2_c, in_filters, k_size, 1, 'conv1_3')
            return self.activation_out(x + x3, act_out)

    def keep_ma(self, x, in_filters, k_size, name, act_out=True):
        """ conv1 -> conv2 -> conv3
        """
        with tf.variable_scope('ma_' + name):
            x1 = self.conv(x, in_filters, k_size, 1, 'conv1_1')
            x2 = self.conv(x + x1, in_filters, k_size * 2, 1, 'conv1_2')
            x3 = self.conv(x + x2, in_filters, k_size, 1, 'conv1_3')
            return self.activation_out(x + x3, act_out)

    def keep_plain(self, x, in_filters, k_size, name, act_out=True):
        """ conv1 -> conv2 -> conv3
        """
        with tf.variable_scope('plain_' + name):
            x1 = self.conv(x, in_filters, k_size, 1, 'conv1_1')
            x2 = self.conv(x1, in_filters, k_size * 2, 1, 'conv1_2')
            x3 = self.conv(x2, in_filters, k_size, 1, 'conv1_3')
            return x3

    def keep_ma_bn(self, x, in_filters, k_size, name, act_out=True):
        """ conv1 -> conv2 -> conv3
        """
        with tf.variable_scope('ma_bn_' + name):
            x1 = self.conv_bn(x, in_filters, k_size, 1, 'conv1_1')
            x2 = self.conv_bn(x + x1, in_filters, k_size * 2, 1, 'conv1_2')
            x3 = self.conv_bn(x + x2, in_filters, k_size, 1, 'conv1_3', False)
            return self.activation_out(x + x3, act_out)

    def keep_ma_pre_bn(self, x, in_filters, k_size, name, act_out=True):
        """ conv1 -> conv2 -> conv3
        """
        with tf.variable_scope('ma_prebn_' + name):
            x1 = self.conv_pre_bn(x, in_filters, k_size, 1, 'conv1_1')
            x2 = self.conv_pre_bn(x + x1, in_filters, k_size * 2, 1, 'conv1_2')
            x3 = self.conv_pre_bn(x + x2, in_filters, k_size, 1, 'conv1_3')
            return x + x3

    def model_ms_tiny3(self, inputs, num_classes, is_training, num_block=[3, 5, 5, 3]):
        """
        """
        end_points = {}

        with tf.variable_scope('ms_tiny3_' + itos(num_block)):
            # root-3200
            net = self.conv(inputs, 64, 40, 2, name='root')
            net = self.pool1d(net, 3, 2, name='max_pool0')

            # block1-800
            for i in range(num_block[0]):
                net = self.ms_keep(net, 64, 3, cat('k1', i))
            net = self.conv(net, 128, 20, 2, name='k1')

            # block2-400
            for i in range(num_block[1]):
                net = self.ms_keep(net, 128, 1, cat('k2', i))
            net = self.conv(net, 128, 15, 2, name='k2')

            # block3-200
            for i in range(num_block[2]):
                net = self.ms_keep(net, 128, 1, cat('k3', i))
            net = self.conv(net, 256, 10, 2, name='k3')

            # block4-100
            for i in range(num_block[3] - 1):
                net = self.ms_keep(net, 256, 1, cat('k4', i))
            net = self.ms_keep(net, 256, 1, cat('k4', num_block[3]), False)

            # 100
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

    def model_sen(self, inputs, num_classes, is_training, keep, num_block=[1, 1, 1, 1]):
        """
        """
        end_points = {}

        with tf.variable_scope('sen_' + itos(num_block)):
            # root-3200
            net = self.conv(inputs, 64, 40, 2, name='root')
            net = self.pool1d(net, 3, 2, name='max_pool0')

            # block1-800
            for i in range(num_block[0]):
                net = keep(net, 64, 3, cat('k1', i))
            net = self.conv(net, 128, 20, 2, name='k1')

            # block2-400
            for i in range(num_block[1]):
                net = keep(net, 128, 1, cat('k2', i))
            net = self.conv(net, 128, 15, 2, name='k2')
            # branch-400
            net_400 = self.pool1d(net, 15, 4, name='max_pool1')

            # block3-200
            for i in range(num_block[2]):
                net = keep(net, 128, 1, cat('k3', i))
            net = self.conv(net, 256, 10, 2, name='k3')
            # branch-200
            net_200 = self.pool1d(net, 10, 2, name='max_pool2')

            # block4-100
            for i in range(num_block[3]):
                net = keep(net, 256, 1, cat('k4', i))
            net_100 = keep(net, 256, 1, cat('k4', num_block[3]), False)

            # 100
            def output(inputs, num_classes, name='logits'):
                net = tf.reduce_sum(inputs, axis=1)
                net = layers.dropout(
                    net, self.dropout, is_training=is_training)
                logits = layers.fully_connected(
                    net, num_classes,
                    biases_initializer=None,
                    weights_initializer=tf.truncated_normal_initializer(
                        stddev=0.01),
                    weights_regularizer=None,
                    activation_fn=None,
                    scope=name)
                return logits

            logits1 = output(net_100, num_classes, 'logits1')
            logits2 = output(net_200, num_classes, 'logits2')
            logits3 = output(net_400, num_classes, 'logits3')

            logits = (logits1 + logits2 + logits3) / 3.0

            return logits, end_points