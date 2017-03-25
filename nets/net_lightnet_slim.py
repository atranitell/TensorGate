import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers
from nets import net

from tensorflow.python.ops import standard_ops


class lightnet_slim(net.Net):

    def __init__(self):
        self.weight_decay = 0.0005

    def arg_scope(self):
        with arg_scope([layers.conv2d],
                       weights_regularizer=layers.sum_regularizer([
                           layers.l1_regularizer(self.weight_decay),
                           layers.l2_regularizer(self.weight_decay)]),
                       weights_initializer=tf.truncated_normal_initializer(stddev=0.05),
                       biases_initializer=tf.constant_initializer(0.1),
                       activation_fn=tf.nn.relu,
                       padding='SAME'):
            with arg_scope([layers.max_pool2d, layers.avg_pool2d], padding='SAME') as arg_sc:
                return arg_sc

    def model(self, images, num_classes, is_training):

        end_points = {}

        net = layers.conv2d(images, 64, [7, 7], 2)
        block_in = layers.max_pool2d(net, [3, 3], 2)

        with tf.variable_scope('block1'):
            with tf.variable_scope('branch_1_0'):
                net = layers.conv2d(block_in, 64, [1, 1], 1)
                net = layers.conv2d(net, 64, [3, 3], 1)
                branch_1_0 = layers.conv2d(net, 256, [1, 1], 1)
            net = tf.concat(axis=3, values=[branch_1_0, block_in])
            block_in = layers.conv2d(net, 128, [1, 1], 1)

        block_in = layers.max_pool2d(block_in, [3, 3], 2)

        with tf.variable_scope('block2'):
            with tf.variable_scope('branch_2_0'):
                net = layers.conv2d(block_in, 64, [1, 1], 1)
                net = layers.conv2d(net, 64, [3, 3], 1)
                branch_2_0 = layers.conv2d(net, 256, [1, 1], 1)
            net = tf.concat(axis=3, values=[branch_2_0, block_in])
            block_in = layers.conv2d(net, 256, [1, 1], 1)

        block_in = layers.max_pool2d(block_in, [3, 3], 2)

        with tf.variable_scope('block3'):
            with tf.variable_scope('branch_3_0'):
                net = layers.conv2d(block_in, 64, [1, 1], 1)
                net = layers.conv2d(net, 64, [3, 3], 1)
                branch_3_0 = layers.conv2d(net, 256, [1, 1], 1)
            net = tf.concat(axis=3, values=[branch_3_0, block_in])
            block_in = layers.conv2d(net, 512, [1, 1], 1)

        block_in = layers.max_pool2d(block_in, [3, 3], 2)

        with tf.variable_scope('block4'):
            with tf.variable_scope('branch_4_0'):
                net = layers.conv2d(block_in, 64, [1, 1], 1)
                net = layers.conv2d(net, 64, [3, 3], 1)
                branch_4_0 = layers.conv2d(net, 256, [1, 1], 1)
            net = tf.concat(axis=3, values=[branch_4_0, block_in])
            block_in = layers.conv2d(net, 1024, [1, 1], 1)

        block_in = layers.avg_pool2d(block_in, [7, 7], 1, padding='VALID')

        logits = layers.fully_connected(
            block_in, num_classes,
            biases_initializer=tf.zeros_initializer(),
            weights_initializer=tf.truncated_normal_initializer(
                stddev=1 / 1024.0),
            weights_regularizer=None,
            activation_fn=None,
            scope='logits')

        return logits, end_points
