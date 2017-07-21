# -*- coding: utf-8 -*-
""" AudioNet for 1D audio classification or regression.
    Author: Kai JIN
    Updated: 2017/07/21
"""
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers
from gate.net import net


class AudioNet(net.Net):

    def arguments_scope(self):
        with arg_scope([]) as sc:
            return sc

    def model(self, inputs, num_classes, is_training):
        self.is_training = is_training
        self.activation_fn = tf.nn.relu
        return self.model18(inputs, num_classes, is_training)

    def conv1d(self, inputs, filters, kernel_size, strides, name=None):
        # input's shape [batchsize, features, channels]
        return tf.layers.conv1d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='SAME',
            activation=self.activation_fn,
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

    def model18(self, inputs, num_classes, is_training):
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

        end_points = {}

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
            net = layers.dropout(net, keep_prob=self.dropout,
                                 is_training=is_training)

            shape = net.get_shape().as_list()
            logits = layers.fully_connected(
                net, num_classes,
                biases_initializer=tf.zeros_initializer(),
                weights_initializer=tf.truncated_normal_initializer(
                    stddev=1.0 / shape[1]),
                weights_regularizer=None,
                activation_fn=None,
                scope='logits')

            return logits, end_points