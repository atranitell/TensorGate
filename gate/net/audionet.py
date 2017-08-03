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
        return self.audionet(inputs, num_classes, is_training, 'sen53')

    def audionet(self, inputs, num_classes, is_training, kind):
        """ a factory corresponding to different experiments
            sen9: 11
            sen17: 1111
            sen29: 2222
            sen53: 3553
            sen53t: 3553-tiny
            sen71: 35113
        """
        if kind == 'sen9':
            return self.model_ms_short(inputs, num_classes, is_training, [1, 1])
        elif kind == 'sen17':
            return self.model_ms(inputs, num_classes, is_training, [1, 1, 1, 1])
        elif kind == 'sen29':
            return self.model_ms(inputs, num_classes, is_training, [2, 2, 2, 2])
        elif kind == 'sen53':
            return self.model_ms_old(inputs, num_classes, is_training, [3, 5, 5, 3])
        elif kind == 'sen71':
            return self.model_ms_old(inputs, num_classes, is_training, [3, 5, 11, 3])
        elif kind == 'sen53t':
            return self.model_ms_tiny(inputs, num_classes, is_training, [3, 5, 5, 3])
        elif kind == 'sen53_res':
            return self.model_ms_res(inputs, num_classes, is_training, [3, 5, 5, 3])
        elif kind == 'sen53_dense':
            return self.model_ms_dense(inputs, num_classes, is_training, [3, 5, 5, 3])
        else:
            raise ValueError('Unknown network.')

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

    def conv_bn(self, x, num_filters, k_size, k_stride, name, act=True):
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
            if act is True:
                net = self.bn(self.activation_fn(net))
            return net

    def activation_out(self, x, act_out):
        if act_out:
            return self.activation_fn(x)
        else:
            return x

    def residual_keep(self, x, in_filters, k_size, name, act_out=True):
        """ conv1 -> conv2 -> conv3
        """
        with tf.variable_scope(name):
            net = self.conv(x, in_filters, k_size, 1, 'conv1_1')
            net = self.conv(net, in_filters, k_size * 2, 1, 'conv1_2')
            net = self.conv(net, in_filters, k_size, 1, 'conv1_3')
            return self.activation_out(x + net, act_out)

    def dense_keep(self, x, in_filters, k_size, name, act_out=True):
        """ conv1 -> conv2 -> conv3
        """
        with tf.variable_scope(name):
            x1 = self.conv(x, in_filters, k_size, 1, 'conv1_1')
            x2 = self.conv(x + x1, in_filters, k_size * 2, 1, 'conv1_2')
            x3 = self.conv(x + x1 + x2, in_filters, k_size, 1, 'conv1_3')
            return self.activation_out(x + x1 + x2 + x3, act_out)

    def ms_keep(self, x, in_filters, k_size, name, act_out=True):
        """ conv1 -> conv2 -> conv3
        """
        with tf.variable_scope(name):
            x1 = self.conv(x, in_filters, k_size, 1, 'conv1_1')
            x2 = self.conv(x + x1, in_filters, k_size * 2, 1, 'conv1_2')
            x3 = self.conv(x + x2, in_filters, k_size, 1, 'conv1_3')
            return self.activation_out(x + x3, act_out)

    def ms_keep_bn(self, x, in_filters, k_size, name, act_out=True):
        """ conv1 -> conv2 -> conv3
        """
        with tf.variable_scope(name):
            x1 = self.conv_bn(x, in_filters, k_size, 1, 'conv1_1')
            x2 = self.conv_bn(x + x1, in_filters, k_size * 2, 1, 'conv1_2')
            x3 = self.conv_bn(x + x2, in_filters, k_size, 1, 'conv1_3')
            return self.activation_out(x + x3, act_out)

    def model_ms(self, inputs, num_classes, is_training, num_block=[3, 5, 5, 3]):
        """ best model for paper
        """
        end_points = {}

        with tf.variable_scope('model_ms3_' + itos(num_block)):
            # root-3200
            net = self.conv(inputs, 64, 40, 2, name='root')
            net = self.pool1d(net, 3, 2, name='max_pool0')

            # block1-800
            for i in range(num_block[0]):
                net = self.ms_keep(net, 64, 20, cat('k1', i))
            net = self.conv(net, 128, 20, 2, name='k1')

            # block2-400
            for i in range(num_block[1]):
                net = self.ms_keep(net, 128, 15, cat('k2', i))
            net = self.conv(net, 256, 15, 2, name='k2')

            # block3-200
            for i in range(num_block[2]):
                net = self.ms_keep(net, 256, 10, cat('k3', i))
            net = self.conv(net, 512, 10, 2, name='k3')

            # block4-100
            for i in range(num_block[3] - 1):
                net = self.ms_keep(net, 512, 5, cat('k4', i))
            net = self.ms_keep(net, 512, 5, cat('k4', num_block[3]), False)

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

    def model_ms_old(self, inputs, num_classes, is_training, num_block=[3, 5, 5, 3]):
        """ best model for paper
        """
        end_points = {}

        with tf.variable_scope('model_ms3_' + itos(num_block)):
            # root-3200
            net = self.conv(inputs, 64, 40, 2, name='root')
            net = self.pool1d(net, 3, 2, name='max_pool0')

            # block1-800
            for i in range(1, num_block[0] + 1):
                net = self.ms_keep(net, 64, 20, cat('k1', i))
            net = self.conv(net, 128, 20, 2, name='k2')

            # block2-400
            for i in range(1, num_block[1] + 1):
                net = self.ms_keep(net, 128, 15, cat('k2', i))
            net = self.conv(net, 256, 15, 2, name='k3')

            # block3-200
            for i in range(1, num_block[2] + 1):
                net = self.ms_keep(net, 256, 10, cat('k3', i))
            net = self.conv(net, 512, 10, 2, name='k4')

            # block4-100
            for i in range(1, num_block[3]):
                net = self.ms_keep(net, 512, 5, cat('k4', i))
            net = self.ms_keep(net, 512, 5, cat('k4', num_block[3] + 1), False)

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

    def model_ms_bn(self, inputs, num_classes, is_training, num_block=[3, 5, 5, 3]):
        """ best model for paper
        """
        end_points = {}

        with tf.variable_scope('ms_bn_' + itos(num_block)):
            # root-3200
            net = self.conv_bn(inputs, 64, 40, 2, name='root')
            net = self.pool1d(net, 3, 2, name='max_pool0')

            # block1-800
            for i in range(num_block[0]):
                net = self.ms_keep_bn(net, 64, 20, cat('k1', i))
            net = self.conv_bn(net, 128, 20, 2, name='k1')

            # block2-400
            for i in range(num_block[1]):
                net = self.ms_keep_bn(net, 128, 15, cat('k2', i))
            net = self.conv_bn(net, 256, 15, 2, name='k2')

            # block3-200
            for i in range(num_block[2]):
                net = self.ms_keep_bn(net, 256, 10, cat('k3', i))
            net = self.conv_bn(net, 512, 10, 2, name='k3')

            # block4-100
            for i in range(num_block[3] - 1):
                net = self.ms_keep_bn(net, 512, 5, cat('k4', i))
            net = self.ms_keep_bn(net, 512, 5, cat('k4', num_block[3]), False)

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

    def model_ms_tiny(self, inputs, num_classes, is_training, num_block=[3, 5, 5, 3]):
        """
        """
        end_points = {}

        with tf.variable_scope('ms_tiny_' + itos(num_block)):
            # root-3200
            net = self.conv(inputs, 64, 40, 2, name='root')
            net = self.pool1d(net, 3, 2, name='max_pool0')

            # block1-800
            for i in range(num_block[0]):
                net = self.ms_keep(net, 64, 3, cat('k1', i))
            net = self.conv(net, 128, 3, 2, name='k1')

            # block2-400
            for i in range(num_block[1]):
                net = self.ms_keep(net, 128, 3, cat('k2', i))
            net = self.conv(net, 256, 3, 2, name='k2')

            # block3-200
            for i in range(num_block[2]):
                net = self.ms_keep(net, 256, 3, cat('k3', i))
            net = self.conv(net, 512, 3, 2, name='k3')

            # block4-100
            for i in range(num_block[3] - 1):
                net = self.ms_keep(net, 512, 3, cat('k4', i))
            net = self.ms_keep(net, 512, 3, cat('k4', num_block[3]), False)

            # 100
            end_points['gap_conv'] = net
            net = tf.reduce_sum(net, axis=1)
            net = layers.dropout(net, self.dropout, is_training=is_training)

            # shape = net.get_shape().as_list()
            logits = layers.fully_connected(
                net, num_classes,
                biases_initializer=None,
                weights_initializer=tf.truncated_normal_initializer(
                    stddev=0.01),
                weights_regularizer=None,
                activation_fn=None,
                scope='logits')

            return logits, end_points

    def model_ms_res(self, inputs, num_classes, is_training, num_block=[3, 5, 5, 3]):
        """ best model for paper
        """
        end_points = {}

        with tf.variable_scope('ms_res_' + itos(num_block)):
            # root-3200
            net = self.conv(inputs, 64, 40, 2, name='root')
            net = self.pool1d(net, 3, 2, name='max_pool0')

            # block1-800
            for i in range(num_block[0]):
                net = self.residual_keep(net, 64, 20, cat('k1', i))
            net = self.conv(net, 128, 20, 2, name='k1')

            # block2-400
            for i in range(num_block[1]):
                net = self.residual_keep(net, 128, 15, cat('k2', i))
            net = self.conv(net, 256, 15, 2, name='k2')

            # block3-200
            for i in range(num_block[2]):
                net = self.residual_keep(net, 256, 10, cat('k3', i))
            net = self.conv(net, 512, 10, 2, name='k3')

            # block4-100
            for i in range(num_block[3] - 1):
                net = self.residual_keep(net, 512, 5, cat('k4', i))
            net = self.residual_keep(
                net, 512, 5, cat('k4', num_block[3]), False)

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

    def model_ms_dense(self, inputs, num_classes, is_training, num_block=[3, 5, 5, 3]):
        """ best model for paper
        """
        end_points = {}

        with tf.variable_scope('ms_dense_' + itos(num_block)):
            # root-3200
            net = self.conv(inputs, 64, 40, 2, name='root')
            net = self.pool1d(net, 3, 2, name='max_pool0')

            # block1-800
            for i in range(num_block[0]):
                net = self.dense_keep(net, 64, 20, cat('k1', i))
            net = self.conv(net, 128, 20, 2, name='k1')

            # block2-400
            for i in range(num_block[1]):
                net = self.dense_keep(net, 128, 15, cat('k2', i))
            net = self.conv(net, 256, 15, 2, name='k2')

            # block3-200
            for i in range(num_block[2]):
                net = self.dense_keep(net, 256, 10, cat('k3', i))
            net = self.conv(net, 512, 10, 2, name='k3')

            # block4-100
            for i in range(num_block[3] - 1):
                net = self.dense_keep(net, 512, 5, cat('k4', i))
            net = self.dense_keep(net, 512, 5, cat('k4', num_block[3]), False)

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

    def model_ms_short(self, inputs, num_classes, is_training, num_block=[1, 1]):
        """ best model for paper
        """
        end_points = {}

        with tf.variable_scope('model_ms_short' + itos(num_block)):
            # root-3200
            net = self.conv(inputs, 64, 40, 2, name='root')
            net = self.pool1d(net, 3, 2, name='max_pool0')

            # block1-800
            for i in range(num_block[0]):
                net = self.ms_keep(net, 64, 20, cat('k1', i))
            net = self.conv(net, 128, 20, 2, name='k1')

            # block2-400
            for i in range(num_block[1] - 1):
                net = self.ms_keep(net, 128, 15, cat('k2', i))
            net = self.ms_keep(net, 128, 15, cat('k2', num_block[1]), False)

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
