
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers
from tensorflow.contrib import metrics
from nets import net


class lenet(net.Net):

    def __init__(self):
        pass

    def arg_scope(self):
        weight_decay = 0.0
        with arg_scope([layers.conv2d, layers.fully_connected],
                       weights_regularizer=layers.l2_regularizer(weight_decay),
                       weights_initializer=tf.truncated_normal_initializer(
                           stddev=0.1),
                       activation_fn=tf.nn.relu) as sc:
            return sc

    def model(self, images, num_classes, is_training):
        end_points = {}
        dropout_keep_prob = 0.9
        with tf.variable_scope('lenet', 'LeNet', [images, num_classes]):
            net = layers.conv2d(images, 32, [5, 5], scope='conv1')
            net = layers.max_pool2d(net, [2, 2], 2, scope='pool1')
            net = layers.conv2d(net, 64, [5, 5], scope='conv2')
            net = layers.max_pool2d(net, [2, 2], 2, scope='pool2')
            net = layers.flatten(net)
            end_points['Flatten'] = net

            net = layers.fully_connected(net, 1024, scope='fc3')
            net = layers.dropout(net, dropout_keep_prob, is_training=is_training,
                                 scope='dropout3')
            logits = layers.fully_connected(net, num_classes, activation_fn=None,
                                            scope='fc4')

        end_points['Logits'] = logits
        end_points['Predictions'] = layers.softmax(logits, scope='Predictions')

        return logits, end_points
