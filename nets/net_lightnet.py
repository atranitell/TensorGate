import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers
from nets import net


class lightnet(net.Net):

    def __init__(self):
        self.weight_decay = 0.0001
        self.batch_norm_decay = 0.997
        self.batch_norm_epsilon = 1e-5
        self.batch_norm_scale = True

    def arg_scope(self):
        weight_decay = self.weight_decay
        batch_norm_decay = self.batch_norm_decay
        batch_norm_epsilon = self.batch_norm_epsilon
        batch_norm_scale = self.batch_norm_scale

        batch_norm_params = {
            'decay': batch_norm_decay,
            'epsilon': batch_norm_epsilon,
            'scale': batch_norm_scale,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
        }

        with arg_scope([layers.conv2d],
                       weights_regularizer=layers.l2_regularizer(weight_decay),
                       weights_initializer=layers.variance_scaling_initializer(),
                       activation_fn=tf.nn.relu,
                       normalizer_fn=layers.batch_norm,
                       normalizer_params=batch_norm_params,
                       padding='SAME'):
            with arg_scope([layers.batch_norm], **batch_norm_params):
                with arg_scope([layers.max_pool2d, layers.avg_pool2d], padding='SAME') as arg_sc:
                    return arg_sc

    def model(self, images, num_classes, is_training):

        end_points = {}

        with arg_scope([layers.batch_norm], is_training=is_training):

            net = layers.conv2d(images, 64, [7, 7], 2)
            block_in = layers.max_pool2d(net, [3, 3], 2)

            with tf.variable_scope('block1'):
                with tf.variable_scope('branch_1_0'):
                    net = layers.conv2d(block_in, 64, [1, 1], 1)
                    net = layers.conv2d(net, 64, [3, 3], 1)
                    branch_1_0 = layers.conv2d(net, 256, [1, 1], 1)
                with tf.variable_scope('branch_1_1'):
                    branch_1_1 = layers.conv2d(block_in, 256, [1, 1], 1)
                with tf.variable_scope('branch_1_2'):
                    net = layers.avg_pool2d(block_in, [3, 3], 1)
                    branch_1_2 = layers.conv2d(net, 256, [1, 1], 1)
                net = tf.concat(axis=3, values=[branch_1_0, branch_1_1, branch_1_2])
                block_in = layers.conv2d(net, 768, [1, 1], 1)

            block_in = layers.avg_pool2d(block_in, [3, 3], 2)

            with tf.variable_scope('block2'):
                with tf.variable_scope('branch_2_0'):
                    net = layers.conv2d(block_in, 64, [1, 1], 1)
                    net = layers.conv2d(net, 64, [3, 3], 1)
                    branch_2_0 = layers.conv2d(net, 256, [1, 1], 1)
                net = tf.concat(axis=3, values=[branch_2_0, block_in])
                block_in = layers.conv2d(net, 1024, [1, 1], 1)

            block_in = layers.avg_pool2d(block_in, [3, 3], 2)

            with tf.variable_scope('block3'):
                with tf.variable_scope('branch_3_0'):
                    net = layers.conv2d(block_in, 64, [1, 1], 1)
                    net = layers.conv2d(net, 64, [3, 3], 1)
                    branch_3_0 = layers.conv2d(net, 256, [1, 1], 1)
                with tf.variable_scope('branch_3_1'):
                    branch_3_1 = layers.conv2d(block_in, 256, [1, 1], 1)
                with tf.variable_scope('branch_3_2'):
                    net = layers.avg_pool2d(block_in, [3, 3], 1)
                    branch_3_2 = layers.conv2d(net, 256, [1, 1], 1)
                net = tf.concat(axis=3, values=[branch_3_0, branch_3_1, branch_3_2])
                block_in = layers.conv2d(net, 768, [1, 1], 1)

            with tf.variable_scope('block4'):
                with tf.variable_scope('branch_4_0'):
                    net = layers.conv2d(block_in, 64, [1, 1], 1)
                    net = layers.conv2d(net, 64, [3, 3], 1)
                    branch_4_0 = layers.conv2d(net, 256, [1, 1], 1)
                net = tf.concat(axis=3, values=[branch_4_0, block_in])
                block_in = layers.conv2d(net, 1024, [1, 1], 1)

            block_in = layers.avg_pool2d(block_in, [3, 3], 2)

            with tf.variable_scope('block5'):
                with tf.variable_scope('branch_5_0'):
                    net = layers.conv2d(block_in, 64, [1, 1], 1)
                    net = layers.conv2d(net, 64, [3, 3], 1)
                    branch_5_0 = layers.conv2d(net, 256, [1, 1], 1)
                with tf.variable_scope('branch_5_1'):
                    branch_5_1 = layers.conv2d(block_in, 256, [1, 1], 1)
                with tf.variable_scope('branch_5_2'):
                    net = layers.avg_pool2d(block_in, [3, 3], 1)
                    branch_5_2 = layers.conv2d(net, 256, [1, 1], 1)
                net = tf.concat(axis=3, values=[branch_5_0, branch_5_1, branch_5_2])
                block_in = layers.conv2d(net, 768, [1, 1], 1)

            with tf.variable_scope('block6'):
                with tf.variable_scope('branch_6_0'):
                    net = layers.conv2d(block_in, 64, [1, 1], 1)
                    net = layers.conv2d(net, 64, [3, 3], 1)
                    branch_6_0 = layers.conv2d(net, 256, [1, 1], 1)
                net = tf.concat(axis=3, values=[branch_6_0, block_in])
                block_in = layers.conv2d(net, 1024, [1, 1], 1)

            block_in = layers.avg_pool2d(block_in, [7, 7], 1, padding='VALID')

            logits = tf.reduce_mean(block_in, [1, 2, 3])

        return logits, end_points
