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
        return self.model_ms(inputs, num_classes, is_training)

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

    def conv(self, x, num_filters, k_size, k_stride, name):
        """ BN->RELU->Weight
        """
        with tf.variable_scope(name):
            # net = self.bn(x)
            # net = self.activation_fn(net)
            net = tf.layers.conv1d(
                inputs=x,
                filters=num_filters,
                kernel_size=k_size,
                strides=k_stride,
                padding='SAME',
                activation=self.activation_fn,
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

    def residual_keep(self, x, in_filters, k_size, name, act_out=True):
        """ conv1 -> conv2 -> conv3
        """
        with tf.variable_scope(name):
            net = self.conv(x, in_filters / 2, k_size, 1, 'conv1_1')
            net = self.conv(net, in_filters / 2, k_size * 2, 1, 'conv1_2')
            net = self.conv(net, in_filters, k_size, 1, 'conv1_3')
            return self.activation_out(x + net, act_out)

    def residual_down(self, x, in_filters, k_size, name, act_out=True):
        """ output filters = 2 * in_filters
            output size = size(x) / 4
        """
        with tf.variable_scope(name):
            # b1
            net = self.conv(x, in_filters / 2, k_size, 2, 'conv1_1')
            net = self.conv(net, in_filters / 2, k_size * 2, 2, 'conv1_2')
            net1 = self.conv(net, in_filters * 2, k_size, 1, 'conv1_3')
            # b2
            net2 = self.conv(x, in_filters * 2, k_size, 4, 'conv2_1')
            return self.activation_out(net1 + net2, act_out)

    def model_res(self, inputs, num_classes, is_training):
        """ add a same conv in the middle of branch1
        """
        end_points = {}

        with tf.variable_scope('model_res135'):
            num_block = [1, 3, 5]

            # root
            net = self.conv(inputs, 64, 40, 2, name='root')

            # residual-1 - 1600
            for i in range(1, num_block[0] + 1):
                net = self.residual_keep(net, 64, 20, cat('res_k1', i))
            net = self.residual_down(net, 64, 20, 'res_d1')

            # residual-2 - 400
            for i in range(1, num_block[1] + 1):
                net = self.residual_keep(net, 128, 10, cat('res_k2', i))
            net = self.residual_down(net, 128, 10, 'res_d2')

            # residual-3 - 100
            for i in range(1, num_block[2] + 1):
                net = self.residual_keep(net, 256, 5, cat('res_k3', i))
            net = self.residual_down(net, 256, 5, 'res_d3', False)

            net = tf.reduce_sum(net, axis=1)
            net = layers.dropout(net, self.dropout, is_training=is_training)

            # shape = net.get_shape().as_list()
            logits = layers.fully_connected(
                net, num_classes,
                biases_initializer=tf.zeros_initializer(),
                weights_initializer=tf.truncated_normal_initializer(
                    stddev=0.01),
                weights_regularizer=None,
                activation_fn=None,
                scope='logits')

            return logits, end_points

    def dense_keep(self, x, in_filters, k_size, name, act_out=True):
        """ conv1 -> conv2 -> conv3
        """
        with tf.variable_scope(name):
            x1 = self.conv(x, in_filters / 2, k_size, 1, 'conv1_1')
            x2_c = tf.concat([x, x1], axis=2)
            x2 = self.conv(x2_c, in_filters / 2, k_size * 2, 1, 'conv1_2')
            x3_c = tf.concat([x, x2], axis=2)
            x3 = self.conv(x3_c, in_filters, k_size, 1, 'conv1_3')
            return self.activation_out(x + x3, act_out)

    def dense_down(self, x, in_filters, k_size, name, act_out=True):
        """ output filters = 2 * in_filters
            output size = size(x) / 4
        """
        with tf.variable_scope(name):
            # b1
            net = self.conv(x, in_filters / 2, k_size, 2, 'conv1_1')
            net = self.conv(net, in_filters / 2, k_size * 2, 2, 'conv1_2')
            net1 = self.conv(net, in_filters * 2, k_size, 1, 'conv1_3')
            # b2
            net2 = self.conv(x, in_filters * 2, k_size, 4, 'conv2_1')
            return self.activation_out(net1 + net2, act_out)

    def model_dense(self, inputs, num_classes, is_training):
        """ add a same conv in the middle of branch1
        """
        end_points = {}
        num_block = [1, 3, 5]

        with tf.variable_scope('model_dense' + itos(num_block)):
            # root
            net = self.conv(inputs, 64, 40, 2, name='root')

            # residual-1 - 1600
            for i in range(1, num_block[0] + 1):
                net = self.dense_keep(net, 64, 20, cat('dense_k1', i))
            net = self.dense_down(net, 64, 20, 'dense_d1')

            # residual-2 - 400
            for i in range(1, num_block[1] + 1):
                net = self.dense_keep(net, 128, 10, cat('dense_k2', i))
            net = self.dense_down(net, 128, 10, 'dense_d2')

            # residual-3 - 100
            for i in range(1, num_block[2] + 1):
                net = self.dense_keep(net, 256, 5, cat('dense_k3', i))
            net = self.dense_down(net, 256, 5, 'dense_d3', False)

            # 25
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

    def ms_keep(self, x, in_filters, k_size, name, act_out=True):
        """ conv1 -> conv2 -> conv3
        """
        with tf.variable_scope(name):
            x1 = self.conv(x, in_filters, k_size, 1, 'conv1_1')
            x2 = self.conv(x + x1, in_filters, k_size * 2, 1, 'conv1_2')
            x3 = self.conv(x + x2, in_filters, k_size, 1, 'conv1_3')
            return self.activation_out(x + x3, act_out)

    def ms_down(self, x, in_filters, k_size, name, act_out=True):
        """ output filters = 2 * in_filters
            output size = size(x) / 4
        """
        with tf.variable_scope(name):
            # b1
            net = self.conv(x, in_filters, k_size, 2, 'conv1_1')
            net = self.conv(net, in_filters, k_size * 2, 2, 'conv1_2')
            net1 = self.conv(net, in_filters * 2, k_size, 1, 'conv1_3')
            # b2
            net2 = self.conv(x, in_filters * 2, k_size, 4, 'conv2_1')
            return self.activation_out(net1 + net2, act_out)

    def model_ms(self, inputs, num_classes, is_training):
        """ If input is:
            12800-> 6400-> 1600-> 400 -> 100
            6400 -> 3200-> 800 -> 200 -> 50
            3200 -> 1600-> 400 -> 100 -> 25
        """
        end_points = {}
        num_block = [3, 5, 3]

        with tf.variable_scope('model_ms_' + itos(num_block)):
            # root
            net = self.conv(inputs, 64, 40, 2, name='root')

            # residual-1 - 1600
            for i in range(1, num_block[0] + 1):
                net = self.ms_keep(net, 64, 20, cat('k1', i))
            net = self.ms_down(net, 64, 20, 'd1')

            # residual-2 - 400
            for i in range(1, num_block[1] + 1):
                net = self.ms_keep(net, 128, 10, cat('k2', i))
            net = self.ms_down(net, 128, 10, 'd2')

            # residual-3 - 100
            for i in range(1, num_block[2] + 1):
                net = self.ms_keep(net, 256, 5, cat('k3', i))
            net = self.ms_down(net, 256, 5, 'd3', False)

            # 25
            end_points['gap_conv'] = net
            net = tf.reduce_sum(net, axis=1)
            # net = layers.dropout(net, self.dropout, is_training=is_training)

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
