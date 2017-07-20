# -*- coding: utf-8 -*-
""" regression task for audo
    updated: 2017/06/17
    1D deep qusi-residual neural network
"""
import os
import math
import time

import tensorflow as tf
from tensorflow.contrib import framework
from tensorflow.contrib import layers
from tensorflow.contrib import slim

import gate
from gate.utils.logger import logger
import gate.dataset.avec.utils as avec2014_error


class resnet_1d():

    def __init__(self):
        self.weight_decay = 0.0005
        self.batch_norm_decay = 0.99
        self.batch_norm_epsilon = 1e-5
        self.batch_norm_scale = True

    def conv1d(self, inputs, filters, kernel_size, strides, name=None, activation_fn=tf.nn.relu):
        # input's shape [batchsize, features, channels]
        return tf.layers.conv1d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='SAME',
            activation=activation_fn,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            kernel_regularizer=layers.l2_regularizer(self.weight_decay),
            bias_initializer=tf.constant_initializer(0.0),
            name=name)

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

    def model4(self, inputs, is_training):
        """ add a same conv in the middle of branch1
        """
        self.is_training = is_training
        with tf.variable_scope('audionet_v1_residual_relu4'):

            with tf.variable_scope('block1'):
                net = self.conv1d(inputs, 64, 1, 1)
                block_in = tf.nn.relu(net)
                with tf.variable_scope('branch1'):
                    net = self.conv1d(block_in, 64, 20, 4)
                    net = tf.nn.relu(net)
                    net = self.conv1d(net, 128, 20, 1)
                    net = tf.nn.relu(net)
                    net = self.conv1d(net, 64, 20, 4)
                    out1 = tf.nn.relu(net)
                with tf.variable_scope('branch2'):
                    out2 = self.pool1d(block_in, 20, 16)
                out = out1 + out2

            with tf.variable_scope('block2'):
                net = self.conv1d(out, 128, 1, 1)
                block_in = tf.nn.relu(net)
                with tf.variable_scope('branch1'):
                    net = self.conv1d(block_in, 128, 10, 2)
                    net = tf.nn.relu(net)
                    net = self.conv1d(net, 256, 10, 1)
                    net = tf.nn.relu(net)
                    net = self.conv1d(net, 128, 10, 2)
                    out1 = tf.nn.relu(net)
                with tf.variable_scope('branch2'):
                    out2 = self.pool1d(block_in, 10, 4)
                out = out1 + out2

            net = tf.reduce_sum(out, [1])
            shape = net.get_shape().as_list()

            logits = layers.fully_connected(
                net, 1,
                biases_initializer=tf.zeros_initializer(),
                weights_initializer=tf.truncated_normal_initializer(
                    stddev=1.0 / shape[1]),
                weights_regularizer=None,
                activation_fn=None,
                scope='logits')

            return logits

    def model11(self, inputs, is_training):
        """ add a same conv in the middle of branch1
        """
        def resblock_1(block_in, in_num_, num_, size_, name='', activation=False):
            with tf.variable_scope(name + '_resblock_1'):
                with tf.variable_scope('branch1'):
                    net = self.conv1d(block_in, num_, size_, 1, 'conv1')
                    net = self.activation_fn(net)
                    net = self.conv1d(net, num_, size_ * 2, 1, 'conv2')
                    net = self.activation_fn(net)
                    net = self.conv1d(net, in_num_, size_, 1, 'conv3')
                    net = self.activation_fn(net)
                if activation:
                    return self.activation_fn(block_in + net)
                else:
                    return block_in + net

        def resblock_2(block_in, in_num_, num_, size_, name='', activation=False):
            with tf.variable_scope(name + '_resblock_2'):
                with tf.variable_scope('branch1'):
                    net = self.conv1d(block_in, num_, size_, 2, 'conv1')
                    net = self.activation_fn(net)
                    net = self.conv1d(net, num_, size_ * 2, 1, 'conv2')
                    net = self.activation_fn(net)
                    net = self.conv1d(net, in_num_, size_, 2, 'conv3')
                    net = self.activation_fn(net)
                with tf.variable_scope('branch2'):
                    out2 = self.pool1d(block_in, size_, 4, name='pool1')
                if activation:
                    return self.activation_fn(out2 + net)
                else:
                    return out2 + net

        self.is_training = is_training
        self.activation_fn = tf.nn.relu

        with tf.variable_scope('audionet_11'):
            num_block = 5  # 2

            net = self.conv1d(inputs, 32, 20, 2, name='conv1')
            net = self.activation_fn(net)
            for i in range(num_block):
                net = resblock_1(
                    net, 32, 16, 20, 'block1_1_' + str(i + 1), True)
            net = resblock_2(net, 32, 16, 20, 'block1_2', True)

            net = self.conv1d(net, 64, 20, 2, name='conv2')
            net = self.activation_fn(net)
            for i in range(num_block):
                net = resblock_1(
                    net, 64, 32, 20, 'block2_1_' + str(i + 1), True)
            net = resblock_2(net, 64, 32, 20, 'block2_2', True)

            net = self.conv1d(net, 128, 10, 2, name='conv3')
            net = self.activation_fn(net)
            for i in range(num_block):
                net = resblock_1(net, 128, 64, 10,
                                 'block3_1_' + str(i + 1), True)
            net = resblock_2(net, 128, 64, 10, 'block3_2', True)

            net = self.conv1d(net, 256, 10, 2, name='conv4')
            net = self.activation_fn(net)
            for i in range(num_block):
                net = resblock_1(net, 256, 128, 10,
                                 'block4_1_' + str(i + 1), True)
            out = resblock_2(net, 256, 128, 10, 'block4_2', False)

            net = tf.reduce_sum(out, [1])
            shape = net.get_shape().as_list()

            logits = layers.fully_connected(
                net, 1,
                biases_initializer=tf.zeros_initializer(),
                weights_initializer=tf.truncated_normal_initializer(
                    stddev=1.0 / shape[1]),
                weights_regularizer=None,
                activation_fn=None,
                scope='logits')

            return logits

    def model12(self, inputs, is_training):
        """ add a same conv in the middle of branch1
        """
        def res1(block_in, in_num_, num_, size_, name='', activation=False):
            with tf.variable_scope(name + '_res1'):
                with tf.variable_scope('b1'):
                    net = self.conv1d(block_in, num_, size_, 1, 'conv1')
                    net = self.activation_fn(net)
                    net = self.conv1d(net, num_, size_ * 2, 1, 'conv2')
                    net = self.activation_fn(net)
                    net = self.conv1d(net, in_num_, size_, 1, 'conv3')
                    net = self.activation_fn(net)
                if activation:
                    return self.activation_fn(block_in + net)
                else:
                    return block_in + net

        def res2(block_in, in_num_, num_, size_, name='', activation=False):
            with tf.variable_scope(name + '_res2'):
                with tf.variable_scope('b1'):
                    net = self.conv1d(block_in, num_, size_, 2, 'conv1')
                    net = self.activation_fn(net)
                    net = self.conv1d(net, num_, size_ * 2, 1, 'conv2')
                    net = self.activation_fn(net)
                    net = self.conv1d(net, in_num_, size_, 2, 'conv3')
                    net = self.activation_fn(net)
                with tf.variable_scope('b2'):
                    out2 = self.pool1d(block_in, size_, 4, name='pool1')
                if activation:
                    return self.activation_fn(out2 + net)
                else:
                    return out2 + net

        self.is_training = is_training
        self.activation_fn = tf.nn.relu

        with tf.variable_scope('audionet_12'):
            num_block = 5  # 2

            net = self.conv1d(inputs, 16, 40, 2, name='conv1')
            net = self.activation_fn(net)
            for i in range(1, num_block + 1):
                net = res1(net, 16, 8, 40, 'block1_' + str(i), True)
            net = res2(net, 16, 8, 40, 'block1', True)

            net = self.conv1d(inputs, 32, 20, 1, name='conv2')
            net = self.activation_fn(net)
            for i in range(1, num_block + 1):
                net = res1(net, 32, 16, 20, 'block2_' + str(i), True)
            net = res2(net, 32, 16, 20, 'block2', True)

            net = self.conv1d(net, 64, 20, 1, name='conv3')
            net = self.activation_fn(net)
            for i in range(1, num_block + 1):
                net = res1(net, 64, 32, 20, 'block3_' + str(i), True)
            net = res2(net, 64, 32, 20, 'block3', True)

            net = self.conv1d(net, 128, 10, 1, name='conv4')
            net = self.activation_fn(net)
            for i in range(1, num_block + 1):
                net = res1(net, 128, 64, 10, 'block4_' + str(i), True)
            net = res2(net, 128, 64, 10, 'block4', True)

            net = self.conv1d(net, 256, 10, 1, name='conv5')
            net = self.activation_fn(net)
            for i in range(1, num_block + 1):
                net = res1(net, 256, 128, 10, 'block5_' + str(i), True)
            out = res2(net, 256, 128, 10, 'block5', False)

            # net = tf.reduce_sum(out, [1])
            net = tf.reduce_mean(out, [1])
            shape = net.get_shape().as_list()

            logits = layers.fully_connected(
                net, 1,
                biases_initializer=tf.zeros_initializer(),
                weights_initializer=tf.truncated_normal_initializer(
                    stddev=1.0 / shape[1]),
                weights_regularizer=None,
                activation_fn=None,
                scope='logits')

            return logits

    def model13(self, inputs, is_training):
        """ add a same conv in the middle of branch1
        """
        def res1(block_in, in_num_, num_, size_, name='', activation=False):
            with tf.variable_scope(name + '_res1'):
                with tf.variable_scope('b1'):
                    net = self.conv1d(block_in, num_, size_, 1, 'conv1')
                    net = self.activation_fn(net)
                    net = self.conv1d(net, num_, size_ * 2, 1, 'conv2')
                    net = self.activation_fn(net)
                    net = self.conv1d(net, in_num_, size_, 1, 'conv3')
                    net = self.activation_fn(net)
                if activation:
                    return self.activation_fn(block_in + net)
                else:
                    return block_in + net

        def res2(block_in, in_num_, num_, size_, name='', activation=False):
            with tf.variable_scope(name + '_res2'):
                with tf.variable_scope('b1'):
                    net = self.conv1d(block_in, num_, size_, 2, 'conv1')
                    net = self.activation_fn(net)
                    net = self.conv1d(net, num_, size_ * 2, 1, 'conv2')
                    net = self.activation_fn(net)
                    net = self.conv1d(net, in_num_, size_, 2, 'conv3')
                    net = self.activation_fn(net)
                with tf.variable_scope('b2'):
                    out2 = self.pool1d(block_in, size_, 4, name='pool1')
                if activation:
                    return self.activation_fn(out2 + net)
                else:
                    return out2 + net

        self.is_training = is_training
        self.activation_fn = tf.nn.relu

        with tf.variable_scope('audionet_13'):
            num_block = 6

            net = self.conv1d(inputs, 16, 40, 2, name='conv1')
            net = self.activation_fn(net)
            for i in range(1, num_block + 1):
                net = res1(net, 16, 8, 40, 'block1_' + str(i), True)
            net = res2(net, 16, 8, 40, 'block1', True)

            net = self.conv1d(net, 32, 20, 1, name='conv2')
            net = self.activation_fn(net)
            for i in range(1, num_block + 1):
                net = res1(net, 32, 16, 20, 'block2_' + str(i), True)
            net = res2(net, 32, 16, 20, 'block2', True)

            net = self.conv1d(net, 64, 20, 2, name='conv3')
            net = self.activation_fn(net)
            for i in range(1, num_block + 1):
                net = res1(net, 64, 32, 20, 'block3_' + str(i), True)
            net = res2(net, 64, 32, 20, 'block3', True)

            net = self.conv1d(net, 128, 10, 1, name='conv4')
            net = self.activation_fn(net)
            for i in range(1, num_block + 1):
                net = res1(net, 128, 64, 10, 'block4_' + str(i), True)
            net = res2(net, 128, 64, 10, 'block4', True)

            net = self.conv1d(net, 256, 10, 1, name='conv5')
            net = self.activation_fn(net)
            for i in range(1, num_block + 1):
                net = res1(net, 256, 128, 10, 'block5_' + str(i), True)
            out = res2(net, 256, 128, 10, 'block5', False)

            # net = tf.reduce_sum(out, [1])
            net = tf.reduce_mean(out, [1])
            shape = net.get_shape().as_list()

            logits = layers.fully_connected(
                net, 1,
                biases_initializer=tf.zeros_initializer(),
                weights_initializer=tf.truncated_normal_initializer(
                    stddev=1.0 / shape[1]),
                weights_regularizer=None,
                activation_fn=None,
                scope='logits')

            return logits

    def model14(self, inputs, is_training):
        """ add a same conv in the middle of branch1
        """
        def res1(block_in, in_num_, num_, size_, name='', activation=False):
            with tf.variable_scope(name + '_r1'):
                with tf.variable_scope('b1'):
                    net = self.conv1d(block_in, num_, size_, 1, 'conv1')
                    net = self.conv1d(net, num_, size_ * 2, 1, 'conv2')
                    net = self.conv1d(net, in_num_, size_, 1, 'conv3')
                if activation:
                    return self.activation_fn(block_in + net, name + '_relu')
                else:
                    return block_in + net

        def res2(block_in, in_num_, num_, size_, name='', activation=False):
            with tf.variable_scope(name + '_r2'):
                with tf.variable_scope('b1'):
                    net = self.conv1d(block_in, num_, size_, 2, 'conv1')
                    net = self.conv1d(net, num_, size_ * 2, 1, 'conv2')
                    net = self.conv1d(net, in_num_, size_, 2, 'conv3')
                with tf.variable_scope('b2'):
                    out2 = self.pool1d(block_in, size_, 4, name='pool1')
                if activation:
                    return self.activation_fn(out2 + net, name=name + '_relu')
                else:
                    return out2 + net

        self.is_training = is_training
        self.activation_fn = tf.nn.relu

        with tf.variable_scope('audionet_14'):
            num_block = [1, 3, 5]

            net = self.conv1d(inputs, 64, 20, 2, name='conv1')
            for i in range(1, num_block[0] + 1):
                net = res1(net, 64, 32, 20, 'block1_' + str(i), True)
            net = res2(net, 64, 32, 20, 'block1', True)

            net = self.conv1d(net, 256, 10, 1, name='conv2')
            for i in range(1, num_block[1] + 1):
                net = res1(net, 256, 128, 10, 'block2_' + str(i), True)
            net = res2(net, 256, 128, 10, 'block2', True)

            net = self.conv1d(net, 512, 5, 1, name='conv3')
            for i in range(1, num_block[2] + 1):
                net = res1(net, 512, 256, 5, 'block3_' + str(i), True)
            out = res2(net, 512, 256, 5, 'block3', False)

            net = tf.reduce_sum(out, axis=1)
            net = layers.dropout(net, keep_prob=0.5, is_training=is_training)

            shape = net.get_shape().as_list()
            logits = layers.fully_connected(
                net, 1,
                biases_initializer=tf.zeros_initializer(),
                weights_initializer=tf.truncated_normal_initializer(
                    stddev=1.0 / shape[1]),
                weights_regularizer=None,
                activation_fn=None,
                scope='logits')

            return logits

    def model15(self, inputs, is_training):
        """ add a same conv in the middle of branch1
        """
        def res1(block_in, in_num_, num_, size_, name='', activation=False):
            with tf.variable_scope(name + '_r1'):
                net = self.conv1d(block_in, num_, size_, 1, 'conv1')
                net = self.conv1d(net, num_, size_ * 2, 1, 'conv2')
                net = self.conv1d(net, in_num_, size_, 1, 'conv3')
                if activation:
                    return tf.nn.relu(block_in + net, name + '_relu')
                else:
                    return block_in + net

        def res2(block_in, in_num_, num_, size_, name='', activation=False):
            with tf.variable_scope(name + '_r2'):
                net = self.conv1d(block_in, num_, size_, 2, 'conv1')
                net = self.conv1d(net, num_, size_ * 2, 1, 'conv2')
                net = self.conv1d(net, in_num_, size_, 2, 'conv3')
                out2 = self.pool1d(block_in, size_, 4, name='pool1')
                if activation:
                    return tf.nn.relu(out2 + net, name=name + '_relu')
                else:
                    return out2 + net

        def module(idx, inputs, num_block, num,
                   size, stride, activation=True):
            idx = str(idx)
            block = 'block' + idx
            net = self.conv1d(inputs, num, size, stride, name='conv' + idx)
            for i in range(1, num_block + 1):
                net = res1(net, num, num / 2, size, block + '_' + str(i), True)
            return res2(net, num, num / 2, size, block, activation)

        self.is_training = is_training

        with tf.variable_scope('audionet_14'):
            num_block = [2, 2, 2]

            with tf.variable_scope('m1'):
                m1_1 = self.pool1d(inputs, 20, 16, name='m1_pool')
                m1_1 = module(1, m1_1, num_block[2], 512, 5, 2, False)

            with tf.variable_scope('m2'):
                m2_1 = self.pool1d(inputs, 20, 4, name='m2_pool')
                m2_1 = module(1, m2_1, num_block[1], 256, 10, 2, False)

            with tf.variable_scope('m3'):
                m3_1 = module(1, inputs, num_block[0], 64, 20, 2)
                m3_2 = module(2, m3_1, num_block[1], 256, 10, 1, False)
                m3_2_add = tf.nn.relu(m3_2 + m2_1, name='m3_2_relu')
                m3_3 = module(3, m3_2_add, num_block[2], 512, 5, 1, False)
                out = m3_3 + m1_1

            net = tf.reduce_sum(out, axis=1)
            net = layers.dropout(net, keep_prob=0.5, is_training=is_training)

            shape = net.get_shape().as_list()
            logits = layers.fully_connected(
                net, 1,
                biases_initializer=tf.zeros_initializer(),
                weights_initializer=tf.truncated_normal_initializer(
                    stddev=1.0 / shape[1]),
                weights_regularizer=None,
                activation_fn=None,
                scope='logits')

            return logits

    def model16(self, inputs, is_training):
        """ add a same conv in the middle of branch1
        """
        def res1(block_in, in_num_, num_, size_, name='', activation=False):
            with tf.variable_scope(name + '_r1'):
                net = self.conv1d(block_in, num_, size_, 1, 'conv1')
                net = self.conv1d(net, num_, size_ * 2, 1, 'conv2')
                net = self.conv1d(net, in_num_, size_, 1, 'conv3')
                if activation:
                    return self.activation_fn(block_in + net, name + '_relu')
                else:
                    return block_in + net

        def res2(block_in, in_num_, num_, size_, name='', activation=False):
            with tf.variable_scope(name + '_r2'):
                net = self.conv1d(block_in, num_, size_, 2, 'conv1')
                net = self.conv1d(net, num_, size_ * 2, 1, 'conv2')
                net = self.conv1d(net, in_num_, size_, 2, 'conv3')
                out2 = self.pool1d(block_in, size_, 4, name='pool1')
                if activation:
                    return self.activation_fn(out2 + net, name=name + '_relu')
                else:
                    return out2 + net

        self.is_training = is_training
        self.activation_fn = tf.nn.relu

        with tf.variable_scope('audionet_16'):
            num_block = [3, 3, 3]

            net = self.conv1d(inputs, 128, 20, 2, name='conv1')
            for i in range(1, num_block[0] + 1):
                net = res1(net, 128, 64, 20, 'block1_' + str(i), True)
            net = res2(net, 128, 64, 20, 'block1', True)

            net = self.conv1d(net, 256, 10, 1, name='conv2')
            for i in range(1, num_block[1] + 1):
                net = res1(net, 256, 128, 10, 'block2_' + str(i), True)
            net = res2(net, 256, 128, 10, 'block2', True)

            net = self.conv1d(net, 512, 5, 1, name='conv3')
            for i in range(1, num_block[2] + 1):
                net = res1(net, 512, 256, 5, 'block3_' + str(i), True)
            out = res2(net, 512, 256, 5, 'block3', False)

            net = tf.reduce_sum(out, axis=1)
            net = layers.dropout(net, keep_prob=0.5, is_training=is_training)

            shape = net.get_shape().as_list()
            logits = layers.fully_connected(
                net, 1,
                biases_initializer=tf.zeros_initializer(),
                weights_initializer=tf.truncated_normal_initializer(
                    stddev=1.0 / shape[1]),
                weights_regularizer=None,
                activation_fn=None,
                scope='logits')

            return logits

    def model17(self, inputs, is_training):
        """ add a same conv in the middle of branch1
        """
        def res1(block_in, in_num_, num_, size_, name='', activation=False):
            with tf.variable_scope(name + '_r1'):
                net = self.conv1d(block_in, num_, size_, 1, 'conv1')
                net = self.conv1d(net, num_, size_ * 2, 1, 'conv2')
                net = self.conv1d(net, in_num_, size_, 1, 'conv3')
                if activation:
                    return self.activation_fn(block_in + net, name + '_relu')
                else:
                    return block_in + net

        def res2(block_in, in_num_, num_, size_, name='', activation=False):
            with tf.variable_scope(name + '_r2'):
                net = self.conv1d(block_in, num_, size_, 2, 'conv1')
                net = self.conv1d(net, num_, size_ * 2, 1, 'conv2')
                net = self.conv1d(net, in_num_, size_, 2, 'conv3')
                out2 = self.pool1d(block_in, size_, 4, name='pool1')
                if activation:
                    return self.activation_fn(out2 + net, name=name + '_relu')
                else:
                    return out2 + net

        self.is_training = is_training
        self.activation_fn = tf.nn.relu

        with tf.variable_scope('audionet_17'):
            num_block = [1, 1]

            net = self.conv1d(inputs, 128, 20, 2, name='conv1')
            for i in range(1, num_block[0] + 1):
                net = res1(net, 128, 64, 20, 'block1_' + str(i), True)
            net = res2(net, 128, 64, 20, 'block1', True)

            net = self.conv1d(net, 256, 10, 1, name='conv2')
            for i in range(1, num_block[1] + 1):
                net = res1(net, 256, 128, 10, 'block2_' + str(i), True)
            net = res2(net, 256, 128, 10, 'block2', False)

            net = tf.reduce_sum(net, axis=1)
            net = layers.dropout(net, keep_prob=0.5, is_training=is_training)

            shape = net.get_shape().as_list()
            logits = layers.fully_connected(
                net, 1,
                biases_initializer=tf.zeros_initializer(),
                weights_initializer=tf.truncated_normal_initializer(
                    stddev=1.0 / shape[1]),
                weights_regularizer=None,
                activation_fn=None,
                scope='logits')

            return logits

    def model18(self, inputs, is_training):
        """ add a same conv in the middle of branch1
        """
        def res1(block_in, in_num_, num_, size_, name='', activation=False):
            with tf.variable_scope(name + '_r1'):
                net = self.conv1d(block_in, num_, size_, 1, 'conv1')
                net = self.conv1d(net, num_, size_ * 2, 1, 'conv2')
                net = self.conv1d(net, in_num_, size_, 1, 'conv3')
                if activation:
                    return self.activation_fn(block_in + net, name + '_relu')
                else:
                    return block_in + net

        def res2(block_in, in_num_, num_, size_, name='', activation=False):
            with tf.variable_scope(name + '_r2'):
                net = self.conv1d(block_in, num_, size_, 2, 'conv1')
                net = self.conv1d(net, num_, size_ * 2, 1, 'conv2')
                net = self.conv1d(net, in_num_, size_, 2, 'conv3')
                out2 = self.pool1d(block_in, size_, 4, name='pool1')
                if activation:
                    return self.activation_fn(out2 + net, name=name + '_relu')
                else:
                    return out2 + net

        self.is_training = is_training
        self.activation_fn = tf.nn.relu

        with tf.variable_scope('audionet_18'):
            num_block = [1, 1, 1]

            net = self.conv1d(inputs, 128, 40, 2, name='conv1')
            for i in range(1, num_block[0] + 1):
                net = res1(net, 128, 64, 20, 'block1_' + str(i), True)
            net = res2(net, 128, 64, 20, 'block1', True)

            net = self.conv1d(net, 256, 20, 1, name='conv2')
            for i in range(1, num_block[1] + 1):
                net = res1(net, 256, 128, 10, 'block2_' + str(i), True)
            net = res2(net, 256, 128, 10, 'block2', True)

            net = self.conv1d(net, 512, 20, 1, name='conv3')
            for i in range(1, num_block[2] + 1):
                net = res1(net, 512, 256, 10, 'block3_' + str(i), True)
            out = res2(net, 512, 256, 10, 'block3', False)

            net = tf.reduce_sum(out, axis=1)
            net = layers.dropout(net, keep_prob=0.5, is_training=is_training)

            shape = net.get_shape().as_list()
            logits = layers.fully_connected(
                net, 1,
                biases_initializer=tf.zeros_initializer(),
                weights_initializer=tf.truncated_normal_initializer(
                    stddev=1.0 / shape[1]),
                weights_regularizer=None,
                activation_fn=None,
                scope='logits')

            return logits

    def model19(self, inputs, is_training):
        """ add a same conv in the middle of branch1
        """
        def res1(block_in, in_num_, num_, size_, name='', activation=False):
            with tf.variable_scope(name + '_r1'):
                net = self.conv1d(block_in, num_, size_, 1, 'conv1')
                net = self.conv1d(net, num_, size_ * 2, 1, 'conv2')
                net = self.conv1d(net, in_num_, size_, 1, 'conv3')
                if activation:
                    return self.activation_fn(block_in + net, name + '_relu')
                else:
                    return block_in + net

        def res2(block_in, in_num_, num_, size_, name='', activation=False):
            with tf.variable_scope(name + '_r2'):
                net = self.conv1d(block_in, num_, size_, 2, 'conv1')
                net = self.conv1d(net, num_, size_ * 2, 1, 'conv2')
                net = self.conv1d(net, in_num_, size_, 2, 'conv3')
                out2 = self.pool1d(block_in, size_, 4, name='pool1')
                if activation:
                    return self.activation_fn(out2 + net, name=name + '_relu')
                else:
                    return out2 + net

        self.is_training = is_training
        self.activation_fn = tf.nn.relu

        with tf.variable_scope('audionet_19'):
            num_block = [0, 0, 0]

            net = self.conv1d(inputs, 128, 128, 2, name='conv1')
            for i in range(1, num_block[0] + 1):
                net = res1(net, 128, 64, 128, 'block1_' + str(i), True)
            net = res2(net, 128, 64, 128, 'block1', True)

            net = self.conv1d(net, 256, 64, 1, name='conv2')
            for i in range(1, num_block[1] + 1):
                net = res1(net, 256, 128, 64, 'block2_' + str(i), True)
            net = res2(net, 256, 128, 64, 'block2', True)

            net = self.conv1d(net, 512, 32, 1, name='conv3')
            for i in range(1, num_block[2] + 1):
                net = res1(net, 512, 256, 32, 'block3_' + str(i), True)
            out = res2(net, 512, 256, 32, 'block3', False)

            net = tf.reduce_sum(out, axis=1)
            net = layers.dropout(net, keep_prob=0.5, is_training=is_training)

            shape = net.get_shape().as_list()
            logits = layers.fully_connected(
                net, 1,
                biases_initializer=tf.zeros_initializer(),
                weights_initializer=tf.truncated_normal_initializer(
                    stddev=1.0 / shape[1]),
                weights_regularizer=None,
                activation_fn=None,
                scope='logits')

            return logits

    def model20(self, inputs, is_training):
        """ add a same conv in the middle of branch1
        """
        def res1(block_in, in_num_, num_, size_, name='', activation=False):
            with tf.variable_scope(name + '_r1'):
                net = self.conv1d(block_in, num_, size_, 1, 'conv1')
                net = self.conv1d(net, num_, size_ * 2, 1, 'conv2')
                net = self.conv1d(net, in_num_, size_, 1, 'conv3')
                if activation:
                    return self.activation_fn(block_in + net, name + '_relu')
                else:
                    return block_in + net

        def res2(block_in, in_num_, num_, size_, name='', activation=False):
            with tf.variable_scope(name + '_r2'):
                net = self.conv1d(block_in, num_, size_, 2, 'conv1')
                net = self.conv1d(net, num_, size_ * 2, 1, 'conv2')
                net = self.conv1d(net, in_num_, size_, 2, 'conv3')
                out2 = self.pool1d(block_in, size_, 4, name='pool1')
                if activation:
                    return self.activation_fn(out2 + net, name=name + '_relu')
                else:
                    return out2 + net

        self.is_training = is_training
        self.activation_fn = tf.nn.relu

        with tf.variable_scope('audionet_20'):
            num_block = [0, 0, 0]

            net = self.conv1d(inputs, 128, 128, 2, name='conv1')
            for i in range(1, num_block[0] + 1):
                net = res1(net, 128, 64, 128, 'block1_' + str(i), True)
            net = res2(net, 128, 64, 128, 'block1', True)

            net = self.conv1d(net, 256, 64, 1, name='conv2')
            for i in range(1, num_block[1] + 1):
                net = res1(net, 256, 128, 64, 'block2_' + str(i), True)
            out = res2(net, 256, 128, 64, 'block2', False)

            net = tf.reduce_sum(out, axis=1)
            net = layers.dropout(net, keep_prob=0.5, is_training=is_training)

            shape = net.get_shape().as_list()
            logits = layers.fully_connected(
                net, 1,
                biases_initializer=tf.zeros_initializer(),
                weights_initializer=tf.truncated_normal_initializer(
                    stddev=1.0 / shape[1]),
                weights_regularizer=None,
                activation_fn=None,
                scope='logits')

            return logits


    def model18_bn(self, inputs, is_training):
        """ add a same conv in the middle of branch1
        """
        def res1(block_in, in_num_, num_, size_, name='', activation=False):
            with tf.variable_scope(name + '_r1'):
                net = self.conv1d(block_in, num_, size_, 1, 'conv1', activation_fn=None)
                net = self.activation_fn(self.bn(net))
                net = self.conv1d(net, num_, size_ * 2, 1, 'conv2', activation_fn=None)
                net = self.activation_fn(self.bn(net))
                net = self.conv1d(net, in_num_, size_, 1, 'conv3', activation_fn=None)
                net = self.activation_fn(self.bn(net))
                if activation:
                    return self.activation_fn(block_in + net, name + '_relu')
                else:
                    return block_in + net

        def res2(block_in, in_num_, num_, size_, name='', activation=False):
            with tf.variable_scope(name + '_r2'):
                net = self.conv1d(block_in, num_, size_, 2, 'conv1', activation_fn=None)
                net = self.activation_fn(self.bn(net))
                net = self.conv1d(net, num_, size_ * 2, 1, 'conv2', activation_fn=None)
                net = self.activation_fn(self.bn(net))
                net = self.conv1d(net, in_num_, size_, 2, 'conv3', activation_fn=None)
                net = self.activation_fn(self.bn(net))
                out2 = self.pool1d(block_in, size_, 4, name='pool1')
                if activation:
                    return self.activation_fn(out2 + net, name=name + '_relu')
                else:
                    return out2 + net

        self.is_training = is_training
        self.activation_fn = tf.nn.relu

        with tf.variable_scope('audionet_18_bn'):
            num_block = [1, 1, 1]

            net = self.conv1d(inputs, 128, 40, 2, name='conv1', activation_fn=None)
            net = self.activation_fn(self.bn(net))
            for i in range(1, num_block[0] + 1):
                net = res1(net, 128, 64, 20, 'block1_' + str(i), True)
            net = res2(net, 128, 64, 20, 'block1', True)

            net = self.conv1d(net, 256, 20, 1, name='conv2', activation_fn=None)
            net = self.activation_fn(self.bn(net))
            for i in range(1, num_block[1] + 1):
                net = res1(net, 256, 128, 10, 'block2_' + str(i), True)
            net = res2(net, 256, 128, 10, 'block2', True)

            net = self.conv1d(net, 512, 20, 1, name='conv3', activation_fn=None)
            net = self.activation_fn(self.bn(net))
            for i in range(1, num_block[2] + 1):
                net = res1(net, 512, 256, 10, 'block3_' + str(i), True)
            out = res2(net, 512, 256, 10, 'block3', False)

            net = tf.reduce_sum(out, axis=1)
            net = layers.dropout(net, keep_prob=0.5, is_training=is_training)

            shape = net.get_shape().as_list()
            logits = layers.fully_connected(
                net, 1,
                biases_initializer=tf.zeros_initializer(),
                weights_initializer=tf.truncated_normal_initializer(
                    stddev=1.0 / shape[1]),
                weights_regularizer=None,
                activation_fn=None,
                scope='logits')

            return logits


def get_network(X, dataset, phase, scope=''):
    """
    X shape transforms->
      from [batchsize, time_step, feature]
      to   [batchsize, time_step*feature]
    ps: to nonoverlap among frames,
      you should define rnn.frame_invl = rnn.frame_length
    """
    X = tf.reshape(X, [dataset.batch_size,
                       dataset.audio.frame_length * dataset.audio.frames, 1])

    net = resnet_1d()
    is_training = True if phase is 'train' else False
    logits = net.model18_bn(X, is_training)

    return logits, None


def get_loss(logits, labels, batch_size, num_classes):
    # get loss
    losses, labels, logits = gate.loss.l2.get_loss(
        logits, labels, num_classes, batch_size)
    # get error
    mae, rmse = gate.loss.l2.get_error(
        logits, labels, num_classes)
    return losses, mae, rmse


def train(data_name, chkp_path=None, exclusions=None):
    """ train cnn model
    Args:
        data_name:
        chkp_path:
        exclusion:
    Return:
        None
    """
    with tf.Graph().as_default():
        # get data model
        dataset = gate.dataset.factory.get_dataset(
            data_name, 'train', chkp_path)

        # build data model
        images, labels, _ = dataset.loads()

        # get global step
        global_step = framework.create_global_step()
        tf.summary.scalar('iter', global_step)

        # get network
        logits, nets = get_network(
            images, dataset, 'train')

        # get loss
        losses, mae, rmse = get_loss(
            logits, labels, dataset.batch_size, dataset.num_classes)

        # get updater
        with tf.name_scope('updater'):
            updater = gate.solver.Updater()
            updater.init_default_updater(
                dataset, global_step, losses, exclusions)
            learning_rate = updater.get_learning_rate()
            restore_saver = updater.get_variables_saver()
            train_op = updater.get_train_op()

        # Check point
        with tf.name_scope('checkpoint'):
            snapshot = gate.solver.Snapshot()
            chkp_hook = snapshot.get_chkp_hook(dataset)
            summary_hook = snapshot.get_summary_hook(dataset)
            summary_test = snapshot.get_summary_test(dataset)

        # Running Info
        class Running_Hook(tf.train.SessionRunHook):

            def __init__(self):
                self.mean_loss, self.duration = 0, 0
                self.mean_mae, self.mean_rmse = 0, 0
                self.best_iter_mae, self.best_mae = 0, 1000
                self.best_iter_rmse, self.best_rmse = 0, 1000

            def before_run(self, run_context):
                self._start_time = time.time()
                return tf.train.SessionRunArgs(
                    [global_step, losses, mae, rmse, learning_rate],
                    feed_dict=None)

            def after_run(self, run_context, run_values):
                # accumulate datas
                cur_iter = run_values.results[0] - 1
                self.mean_loss += run_values.results[1]
                self.mean_mae += run_values.results[2]
                self.mean_rmse += run_values.results[3]
                self.duration += (time.time() - self._start_time)

                # print information
                if cur_iter % dataset.log.print_frequency == 0:
                    _invl = dataset.log.print_frequency
                    _loss = self.mean_loss / _invl
                    _mae = self.mean_mae / _invl
                    _rmse = self.mean_rmse / _invl
                    _lr = str(run_values.results[4])
                    _duration = self.duration * 1000 / _invl

                    f_str = gate.utils.string.format_iter(cur_iter)
                    f_str.add('loss', _loss, float)
                    f_str.add('mae', _mae, float)
                    f_str.add('rmse', _rmse, float)
                    f_str.add('lr', _lr)
                    f_str.add('time', _duration, float)
                    logger.train(f_str.get())

                    # set zero
                    self.mean_mae, self.mean_rmse = 0, 0
                    self.mean_loss, self.duration = 0, 0

                # evaluation
                if cur_iter % dataset.log.test_interval == 0 and cur_iter != 0:
                    test_start_time = time.time()
                    test_mae, test_rmse = test(
                        data_name, dataset.log.train_dir, summary_test)
                    test_duration = time.time() - test_start_time

                    val_mae, val_rmse = val(
                        data_name, dataset.log.train_dir, summary_test)

                    if val_mae < self.best_mae:
                        self.best_mae = val_mae
                        self.best_iter_mae = cur_iter
                    if val_rmse < self.best_rmse:
                        self.best_rmse = val_rmse
                        self.best_iter_rmse = cur_iter

                    f_str = gate.utils.string.format_iter(cur_iter)
                    f_str.add('best mae', self.best_mae, float)
                    f_str.add('in', self.best_iter_mae, int)
                    f_str.add('best rmse', self.best_rmse, float)
                    f_str.add('in', self.best_iter_rmse, int)
                    f_str.add('time', test_duration, float)
                    logger.train(f_str.get())

        # record running information
        running_hook = Running_Hook()

        # Start to train
        with tf.train.MonitoredTrainingSession(
                hooks=[chkp_hook, summary_hook, running_hook,
                       tf.train.NanTensorHook(losses)],
                config=tf.ConfigProto(allow_soft_placement=True),
                save_checkpoint_secs=None,
                save_summaries_steps=None) as mon_sess:

            if chkp_path is not None:
                snapshot.restore(mon_sess, chkp_path, restore_saver)

            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def val(data_name, chkp_path, summary_writer=None):
    """ test for regression net
    """
    with tf.Graph().as_default():
        # get dataset
        dataset = gate.dataset.factory.get_dataset(
            data_name, 'val', chkp_path)

        # load data
        images, labels, filenames = dataset.loads()

        # get network
        logits, nets = get_network(images, dataset, 'val')

        # get loss
        losses, mae, rmse = get_loss(
            logits, labels, dataset.batch_size, dataset.num_classes)

        # get saver
        saver = tf.train.Saver(name='restore_all_val')

        with tf.Session() as sess:
            # get latest checkpoint
            snapshot = gate.solver.Snapshot()
            global_step = snapshot.restore(sess, chkp_path, saver)

            # start queue from runner
            coord = tf.train.Coordinator()
            threads = []
            for queuerunner in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(queuerunner.create_threads(
                    sess, coord=coord, daemon=True, start=True))

            # Initial some variables
            num_iter = int(math.ceil(dataset.total_num / dataset.batch_size))
            mean_mae, mean_rmse, mean_loss = 0, 0, 0

            # output to file
            tab = tf.constant(' ', shape=[dataset.batch_size])
            labels_str = tf.as_string(tf.reshape(
                labels, shape=[dataset.batch_size]))
            logits_str = tf.as_string(tf.reshape(
                logits * dataset.num_classes, shape=[dataset.batch_size]))
            test_batch_info = filenames + tab + labels_str + tab + logits_str

            test_info_path = os.path.join(
                dataset.log.val_dir, '%s.txt' % global_step)
            test_info_fp = open(test_info_path, 'wb')

            for _ in range(num_iter):
                # if ctrl-c
                if coord.should_stop():
                    break

                # running session to acuqire value
                feeds = [losses, mae, rmse, test_batch_info]
                _loss, _mae, _rmse, _info = sess.run(feeds)
                mean_loss += _loss
                mean_mae += _mae
                mean_rmse += _rmse

                # save tensor info to text file
                for _line in _info:
                    test_info_fp.write(_line + b'\r\n')

            # stop
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)
            test_info_fp.close()

            # statistic
            mean_loss = 1.0 * mean_loss / num_iter
            mean_rmse = 1.0 * mean_rmse / num_iter
            mean_mae = 1.0 * mean_mae / num_iter

            # output result
            f_str = gate.utils.string.format_iter(global_step)
            f_str.add('total sample', dataset.total_num, int)
            f_str.add('num batch', num_iter, int)
            f_str.add('loss', mean_loss, float)
            f_str.add('mae', mean_mae, float)
            f_str.add('rmse', mean_rmse, float)
            logger.test(f_str.get())

            # for specify dataset
            mean_mae, mean_rmse = avec2014_error.get_accurate_from_file(
                test_info_path, 'img')
            f_str = gate.utils.string.format_iter(global_step)
            f_str.add('loss', mean_loss, float)
            f_str.add('video_mae', mean_mae, float)
            f_str.add('video_rmse', mean_rmse, float)
            logger.val(f_str.get())

            # write to summary
            if summary_writer is not None:
                summary = tf.Summary()
                summary.value.add(
                    tag='val/iter', simple_value=int(global_step))
                summary.value.add(tag='val/mae', simple_value=mean_mae)
                summary.value.add(tag='val/rmse', simple_value=mean_rmse)
                summary.value.add(tag='val/loss', simple_value=mean_loss)
                summary_writer.add_summary(summary, global_step)

            return mean_mae, mean_rmse


def test(data_name, chkp_path, summary_writer=None):
    """ test for regression net
    """
    with tf.Graph().as_default():
        # get dataset
        dataset = gate.dataset.factory.get_dataset(
            data_name, 'test', chkp_path)

        # load data
        images, labels, filenames = dataset.loads()

        # get network
        logits, nets = get_network(images, dataset, 'test')

        # get loss
        losses, mae, rmse = get_loss(
            logits, labels, dataset.batch_size, dataset.num_classes)

        # get saver
        saver = tf.train.Saver(name='restore_all_test')

        with tf.Session() as sess:
            # get latest checkpoint
            snapshot = gate.solver.Snapshot()
            global_step = snapshot.restore(sess, chkp_path, saver)

            # start queue from runner
            coord = tf.train.Coordinator()
            threads = []
            for queuerunner in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(queuerunner.create_threads(
                    sess, coord=coord, daemon=True, start=True))

            # Initial some variables
            num_iter = int(math.ceil(dataset.total_num / dataset.batch_size))
            mean_mae, mean_rmse, mean_loss = 0, 0, 0

            # output to file
            tab = tf.constant(' ', shape=[dataset.batch_size])
            labels_str = tf.as_string(tf.reshape(
                labels, shape=[dataset.batch_size]))
            logits_str = tf.as_string(tf.reshape(
                logits * dataset.num_classes, shape=[dataset.batch_size]))
            test_batch_info = filenames + tab + labels_str + tab + logits_str

            test_info_path = os.path.join(
                dataset.log.test_dir, '%s.txt' % global_step)
            test_info_fp = open(test_info_path, 'wb')

            for _ in range(num_iter):
                # if ctrl-c
                if coord.should_stop():
                    break

                # running session to acuqire value
                feeds = [losses, mae, rmse, test_batch_info]
                _loss, _mae, _rmse, _info = sess.run(feeds)
                mean_loss += _loss
                mean_mae += _mae
                mean_rmse += _rmse

                # save tensor info to text file
                for _line in _info:
                    test_info_fp.write(_line + b'\r\n')

            # stop
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)
            test_info_fp.close()

            # statistic
            mean_loss = 1.0 * mean_loss / num_iter
            mean_rmse = 1.0 * mean_rmse / num_iter
            mean_mae = 1.0 * mean_mae / num_iter

            # output result
            f_str = gate.utils.string.format_iter(global_step)
            f_str.add('total sample', dataset.total_num, int)
            f_str.add('num batch', num_iter, int)
            f_str.add('loss', mean_loss, float)
            f_str.add('mae', mean_mae, float)
            f_str.add('rmse', mean_rmse, float)
            logger.test(f_str.get())

            # for specify dataset
            # it use different compute method for mae/rmse
            # rewrite the mean_x value
            mean_mae, mean_rmse = avec2014_error.get_accurate_from_file(
                test_info_path, 'img')
            f_str = gate.utils.string.format_iter(global_step)
            f_str.add('loss', mean_loss, float)
            f_str.add('video_mae', mean_mae, float)
            f_str.add('video_rmse', mean_rmse, float)
            logger.test(f_str.get())

            # write to summary
            if summary_writer is not None:
                summary = tf.Summary()
                summary.value.add(
                    tag='test/iter', simple_value=int(global_step))
                summary.value.add(tag='test/mae', simple_value=mean_mae)
                summary.value.add(tag='test/rmse', simple_value=mean_rmse)
                summary.value.add(tag='test/loss', simple_value=mean_loss)
                summary_writer.add_summary(summary, global_step)

            return mean_mae, mean_rmse
