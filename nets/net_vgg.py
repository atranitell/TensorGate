"""Contains model definitions for versions of the Oxford VGG network.

These model definitions were introduced in the following technical report:

    Very Deep Convolutional Networks For Large-Scale Image Recognition
    Karen Simonyan and Andrew Zisserman
    arXiv technical report, 2015
    PDF: http://arxiv.org/pdf/1409.1556.pdf
    ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
    CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/
"""
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers
from tensorflow.contrib.layers.python.layers import utils

from nets import net

class vgg(net.Net):

    def __init__(self):
        self.dropout_keep_prob = 1
        self.spatial_squeeze = True
        self.weight_decay = 0.0005

    def arg_scope(self):
        """ weight_decay: The l2 regularization coefficient. """
        weight_decay = self.weight_decay

        with arg_scope(
            [layers.conv2d, layers.fully_connected],
            activation_fn=tf.nn.relu,
            weights_regularizer=layers.l2_regularizer(weight_decay),
                biases_initializer=tf.zeros_initializer()):
            with arg_scope([layers.conv2d], padding='SAME') as arg_sc:
                return arg_sc


class vgg_a(vgg):

    def model(self, images, num_classes, is_training):
        """Oxford Net VGG 11-Layers version A Example.

        Note: All the fully_connected layers have been transformed to conv2d layers.
                To use in classification mode, resize input to 224x224.

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
        dropout_keep_prob = self.dropout_keep_prob
        spatial_squeeze = self.spatial_squeeze

        with tf.variable_scope('vgg_a', 'vgg_a', [images]) as sc:
            end_points_collection = sc.name + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with arg_scope([layers.conv2d, layers.max_pool2d],
                                outputs_collections=end_points_collection):
                net = layers.repeat(images, 1, layers.conv2d,
                                    64, [3, 3], scope='conv1')
                net = layers.max_pool2d(net, [2, 2], scope='pool1')
                net = layers.repeat(net, 1, layers.conv2d, 128, [3, 3], scope='conv2')
                net = layers.max_pool2d(net, [2, 2], scope='pool2')
                net = layers.repeat(net, 2, layers.conv2d, 256, [3, 3], scope='conv3')
                net = layers.max_pool2d(net, [2, 2], scope='pool3')
                net = layers.repeat(net, 2, layers.conv2d, 512, [3, 3], scope='conv4')
                net = layers.max_pool2d(net, [2, 2], scope='pool4')
                net = layers.repeat(net, 2, layers.conv2d, 512, [3, 3], scope='conv5')
                net = layers.max_pool2d(net, [2, 2], scope='pool5')
                # Use conv2d instead of fully_connected layers.
                net = layers.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
                net = layers.dropout(net, dropout_keep_prob, is_training=is_training,
                                    scope='dropout6')
                net = layers.conv2d(net, 4096, [1, 1], scope='fc7')
                net = layers.dropout(net, dropout_keep_prob, is_training=is_training,
                                    scope='dropout7')
                net = layers.conv2d(net, num_classes, [1, 1],
                                    activation_fn=None,
                                    normalizer_fn=None,
                                    scope='fc8')
                # Convert end_points_collection into a end_point dict.
                end_points = utils.convert_collection_to_dict(
                    end_points_collection)
                if spatial_squeeze:
                    net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                    end_points[sc.name + '/fc8'] = net
                return net, end_points


class vgg_16(vgg):

    def model(self, images, num_classes, is_training):
        """Oxford Net VGG 16-Layers version D Example.

        Note: All the fully_connected layers have been transformed to conv2d layers.
                To use in classification mode, resize input to 224x224.

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
        dropout_keep_prob = self.dropout_keep_prob
        spatial_squeeze = self.spatial_squeeze

        with tf.variable_scope('vgg_16', 'vgg_16', [images]) as sc:
            end_points_collection = sc.name + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with arg_scope([layers.conv2d, layers.fully_connected, layers.max_pool2d],
                                outputs_collections=end_points_collection):
                net = layers.repeat(images, 2, layers.conv2d,
                                64, [3, 3], scope='conv1')
                net = layers.max_pool2d(net, [2, 2], scope='pool1')
                net = layers.repeat(net, 2, layers.conv2d, 128, [3, 3], scope='conv2')
                net = layers.max_pool2d(net, [2, 2], scope='pool2')
                net = layers.repeat(net, 3, layers.conv2d, 256, [3, 3], scope='conv3')
                net = layers.max_pool2d(net, [2, 2], scope='pool3')
                net = layers.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv4')
                net = layers.max_pool2d(net, [2, 2], scope='pool4')
                net = layers.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv5')
                net = layers.max_pool2d(net, [2, 2], scope='pool5')
                # Use conv2d instead of fully_connected layers.
                net = layers.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
                net = layers.dropout(net, dropout_keep_prob, is_training=is_training,
                                scope='dropout6')
                net = layers.conv2d(net, 4096, [1, 1], scope='fc7')
                net = layers.dropout(net, dropout_keep_prob, is_training=is_training,
                                scope='dropout7')
                net = layers.conv2d(net, num_classes, [1, 1],
                                activation_fn=None,
                                normalizer_fn=None,
                                scope='fc8')
                # Convert end_points_collection into a end_point dict.
                end_points = utils.convert_collection_to_dict(
                    end_points_collection)
                if spatial_squeeze:
                    net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                    end_points[sc.name + '/fc8'] = net
                return net, end_points


class vgg_19(vgg):

    def model(self, images, num_classes, is_training):
        """Oxford Net VGG 19-Layers version E Example.

        Note: All the fully_connected layers have been transformed to conv2d layers.
                To use in classification mode, resize input to 224x224.

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
        dropout_keep_prob = self.dropout_keep_prob
        spatial_squeeze = self.spatial_squeeze

        with tf.variable_scope('vgg_19', 'vgg_19', [images]) as sc:
            end_points_collection = sc.name + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with arg_scope([layers.conv2d, layers.fully_connected, layers.max_pool2d],
                                outputs_collections=end_points_collection):
                net = layers.repeat(images, 2, layers.conv2d, 64, [3, 3], scope='conv1')
                net = layers.max_pool2d(net, [2, 2], scope='pool1')
                net = layers.repeat(net, 2, layers.conv2d, 128,
                                  [3, 3], scope='conv2')
                net = layers.max_pool2d(net, [2, 2], scope='pool2')
                net = layers.repeat(net, 4, layers.conv2d, 256,
                                  [3, 3], scope='conv3')
                net = layers.max_pool2d(net, [2, 2], scope='pool3')
                net = layers.repeat(net, 4, layers.conv2d, 512,
                                  [3, 3], scope='conv4')
                net = layers.max_pool2d(net, [2, 2], scope='pool4')
                net = layers.repeat(net, 4, layers.conv2d, 512,
                                  [3, 3], scope='conv5')
                net = layers.max_pool2d(net, [2, 2], scope='pool5')
                # Use conv2d instead of fully_connected layers.
                net = layers.conv2d(
                    net, 4096, [7, 7], padding='VALID', scope='fc6')
                net = layers.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout6')
                net = layers.conv2d(net, 4096, [1, 1], scope='fc7')
                net = layers.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout7')
                net = layers.conv2d(net, num_classes, [1, 1],
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  scope='fc8')
                # Convert end_points_collection into a end_point dict.
                end_points = utils.convert_collection_to_dict(
                    end_points_collection)
                if spatial_squeeze:
                    net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                    end_points[sc.name + '/fc8'] = net
                return net, end_points