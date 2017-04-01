import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers
from gate.net import net


class lightnet_fast(net.Net):

    def __init__(self):
        self.weight_decay = 0.0001

    def arg_scope(self):
        with arg_scope([layers.conv2d],
                       weights_regularizer=layers.l2_regularizer(
                           self.weight_decay),
                       weights_initializer=tf.truncated_normal_initializer(
                           stddev=0.05),
                       biases_initializer=tf.constant_initializer(0.1),
                       activation_fn=tf.nn.relu,
                       padding='SAME'):
            with arg_scope([layers.max_pool2d, layers.avg_pool2d], padding='SAME') as arg_sc:
                return arg_sc

    def model(self, images, num_classes, is_training):

        end_points = {}

        with tf.variable_scope('lightnet_fast'):

            block_in = layers.conv2d(images, 64, [7, 7], 2)
            block_in = layers.max_pool2d(block_in, [3, 3], 2)

            with tf.variable_scope('block1'):
                with tf.variable_scope('branch_1_0'):
                    _net = layers.conv2d(block_in, 64, [1, 1], 1)
                    _net = layers.conv2d(_net, 64, [3, 3], 1)
                    _net = layers.conv2d(_net, 64, [1, 1], 1)

            block_in = block_in + _net
            block_in = layers.max_pool2d(block_in, [3, 3], 2)

            with tf.variable_scope('block2'):
                with tf.variable_scope('branch_2_0'):
                    _net = layers.conv2d(block_in, 64, [1, 1], 1)
                    _net = layers.conv2d(_net, 64, [3, 3], 1)
                    _net = layers.conv2d(_net, 64, [1, 1], 1)

            block_in = block_in + _net
            block_in = layers.max_pool2d(block_in, [3, 3], 2)

            with tf.variable_scope('block3'):
                with tf.variable_scope('branch_3_0'):
                    _net = layers.conv2d(block_in, 64, [1, 1], 1)
                    _net = layers.conv2d(_net, 64, [3, 3], 1)
                    _net = layers.conv2d(_net, 64, [1, 1], 1)

            block_in = block_in + _net
            block_in = layers.max_pool2d(block_in, [3, 3], 2)

            with tf.variable_scope('block4'):
                with tf.variable_scope('branch_4_0'):
                    _net = layers.conv2d(block_in, 64, [1, 1], 1)
                    _net = layers.conv2d(_net, 64, [3, 3], 1)
                    _net = layers.conv2d(_net, 64, [1, 1], 1)

            block_in = block_in + _net
            block_in = layers.avg_pool2d(block_in, [7, 7], 1, padding='VALID')

            end_points['end_avg_pool'] = block_in

            logits = layers.fully_connected(
                block_in, num_classes,
                biases_initializer=tf.zeros_initializer(),
                weights_initializer=tf.truncated_normal_initializer(
                    stddev=1 / 64.0),
                weights_regularizer=None,
                activation_fn=None,
                scope='logits')

            end_points['logits'] = logits

            return logits, end_points
