"""AlexNet version 2.
Described in: http://arxiv.org/pdf/1404.5997v2.pdf
Parameters from:
github.com/akrizhevsky/cuda-convnet2/blob/master/layers/
layers-imagenet-1gpu.cfg

Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224. To use in fully
        convolutional mode, set spatial_squeeze to false.
        The LRN layers have been removed and change the initializers from
        random_normal_initializer to xavier_initializer.

Args:
    images: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
    layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
    outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.

Returns:
    the last op containing the log predictions and end_points dict.
"""


import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers
from tensorflow.contrib.layers.python.layers import utils
from gate.net import net


class alexnet(net.Net):

    def __init__(self):
        self.weight_decay = 0.0005
        self.dropout_keep_prob = 0.5
        self.spatial_squeeze = True

    def arg_scope(self):
        weight_decay = self.weight_decay

        with arg_scope([layers.conv2d, layers.fully_connected],
                       activation_fn=tf.nn.relu,
                       biases_initializer=tf.constant_initializer(0.1),
                       weights_regularizer=layers.l2_regularizer(weight_decay)):
            with arg_scope([layers.conv2d], padding='SAME'):
                with arg_scope([layers.max_pool2d], padding='VALID') as arg_sc:
                    return arg_sc

    def model(self, images, num_classes, is_training):
        spatial_squeeze = self.spatial_squeeze
        dropout_keep_prob = self.dropout_keep_prob

        end_points = {}

        with tf.variable_scope('alexnet', 'alexnet_v2', [images]) as sc:
            end_points_collection = sc.name + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with arg_scope([layers.conv2d, layers.fully_connected, layers.max_pool2d],
                           outputs_collections=[end_points_collection]):
                net = layers.conv2d(images, 64, [11, 11], 4, padding='VALID', scope='conv1')
                net = layers.max_pool2d(net, [3, 3], 2, scope='pool1')
                net = layers.conv2d(net, 192, [5, 5], scope='conv2')
                net = layers.max_pool2d(net, [3, 3], 2, scope='pool2')
                net = layers.conv2d(net, 384, [3, 3], scope='conv3')
                net = layers.conv2d(net, 384, [3, 3], scope='conv4')
                net = layers.conv2d(net, 256, [3, 3], scope='conv5')
                end_points['end_conv'] = net
                net = layers.max_pool2d(net, [3, 3], 2, scope='pool5')
                end_points['end_avg_pool'] = net
                # Use conv2d instead of fully_connected layers.
                with arg_scope([layers.conv2d],
                               weights_initializer=tf.truncated_normal_initializer(stddev=0.005),
                               biases_initializer=tf.constant_initializer(0.1)):
                    net = layers.conv2d(net, 4096, [5, 5], padding='VALID', scope='fc6')
                    net = layers.dropout(net, dropout_keep_prob,
                                         is_training=is_training, scope='dropout6')
                    net = layers.conv2d(net, 4096, [1, 1], scope='fc7')
                    net = layers.dropout(net, dropout_keep_prob,
                                         is_training=is_training, scope='dropout7')
                    net = layers.conv2d(net, num_classes, [1, 1],
                                        activation_fn=None,
                                        normalizer_fn=None,
                                        biases_initializer=tf.zeros_initializer(),
                                        scope='fc8')

                # Convert end_points_collection into a end_point dict.
                # end_points = utils.convert_collection_to_dict(
                #     end_points_collection)

                if spatial_squeeze:
                    net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                    end_points[sc.name + '/fc8'] = net

                return net, end_points



class alexnet_gap(net.Net):

    def __init__(self):
        self.weight_decay = 0.0005
        self.dropout_keep_prob = 0.5
        self.spatial_squeeze = True

    def arg_scope(self):
        weight_decay = self.weight_decay

        with arg_scope([layers.conv2d, layers.fully_connected],
                       activation_fn=tf.nn.relu,
                       biases_initializer=tf.constant_initializer(0.1),
                       weights_regularizer=layers.l2_regularizer(weight_decay)):
            with arg_scope([layers.conv2d], padding='SAME'):
                with arg_scope([layers.max_pool2d], padding='VALID') as arg_sc:
                    return arg_sc

    def model(self, images, num_classes, is_training):
        spatial_squeeze = self.spatial_squeeze
        dropout_keep_prob = self.dropout_keep_prob

        end_points = {}

        with tf.variable_scope('alexnet', 'alexnet_v2', [images]) as sc:
            end_points_collection = sc.name + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with arg_scope([layers.conv2d, layers.fully_connected, layers.max_pool2d],
                           outputs_collections=[end_points_collection]):
                net = layers.conv2d(images, 64, [11, 11], 4, padding='VALID', scope='conv1')
                net = layers.max_pool2d(net, [3, 3], 2, scope='pool1')
                net = layers.conv2d(net, 192, [5, 5], scope='conv2')
                net = layers.max_pool2d(net, [3, 3], 2, scope='pool2')
                net = layers.conv2d(net, 384, [3, 3], scope='conv3')
                net = layers.conv2d(net, 384, [3, 3], scope='conv4')
                net = layers.conv2d(net, 256, [3, 3], scope='conv5')
                end_points['end_conv'] = net
                net = layers.max_pool2d(net, [3, 3], 2, scope='pool5')
                end_points['end_avg_pool'] = net
                # Use conv2d instead of fully_connected layers.
                with arg_scope([layers.conv2d],
                               weights_initializer=tf.truncated_normal_initializer(stddev=0.005),
                               biases_initializer=tf.constant_initializer(0.1)):
                    # net = layers.conv2d(net, 4096, [5, 5], padding='VALID', scope='fc6')
                    end_points['end_conv'] = net
                    net = layers.avg_pool2d(net, [5, 5], 1, padding='VALID', scope='avg_pool')
                    end_points['end_avg_pool'] = net
                    net = layers.dropout(net, dropout_keep_prob,
                                         is_training=is_training, scope='dropout6')
                    net = layers.conv2d(net, 4096, [1, 1], scope='fc7')
                    net = layers.dropout(net, dropout_keep_prob,
                                         is_training=is_training, scope='dropout7')
                    net = layers.conv2d(net, num_classes, [1, 1],
                                        activation_fn=None,
                                        normalizer_fn=None,
                                        biases_initializer=tf.zeros_initializer(),
                                        scope='fc8')

                # Convert end_points_collection into a end_point dict.
                # end_points = utils.convert_collection_to_dict(
                #     end_points_collection)

                if spatial_squeeze:
                    net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                    end_points[sc.name + '/fc8'] = net

                return net, end_points