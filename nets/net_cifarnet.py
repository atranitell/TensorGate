"""Creates a variant of the CifarNet model.

Note that since the output is a set of 'logits', the values fall in the
interval of (-infinity, infinity). Consequently, to convert the outputs to a
probability distribution over the characters, one will need to convert them
using the softmax function:

        logits = cifarnet.cifarnet(images, is_training=False)
        probabilities = tf.nn.softmax(logits)
        predictions = tf.argmax(logits, 1)

Args:
    images: A batch of `Tensors` of size [batch_size, height, width, channels].
    num_classes: the number of classes in the dataset.
    is_training: specifies whether or not we're currently training the model.
    This variable will determine the behaviour of the dropout layer.
    dropout_keep_prob: the percentage of activation values that are retained.
    prediction_fn: a function to get predictions out of logits.
    scope: Optional variable_scope.

Returns:
    logits: the pre-softmax activations, a tensor of size
    [batch_size, `num_classes`]
    end_points: a dictionary from components of the network to the corresponding
    activation.
"""


import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers
from nets import net


class cifarnet(net.Net):

    def __init__(self):
        self.dropout_keep_prob = 1.0
        self.weight_decay = 0.0001

    def arg_scope(self):
        weight_decay = self.weight_decay

        with arg_scope([layers.conv2d],
                       weights_initializer=tf.truncated_normal_initializer(
                           stddev=5e-2),
                       activation_fn=tf.nn.relu):
            with arg_scope([layers.fully_connected],
                           biases_initializer=tf.constant_initializer(0.1),
                           weights_initializer=tf.truncated_normal_initializer(
                               stddev=0.04),
                           weights_regularizer=layers.l2_regularizer(
                               weight_decay),
                           activation_fn=tf.nn.relu) as sc:
                return sc

    def model(self, images, num_classes, is_training=True):
        end_points = {}
        dropout_keep_prob = self.dropout_keep_prob

        with tf.variable_scope('cifarnet', 'cifarnet', [images, num_classes]):

            net = layers.conv2d(images, 64, [5, 5], scope='conv1')
            end_points['conv1'] = net

            net = layers.max_pool2d(net, [2, 2], 2, scope='pool1')
            end_points['pool1'] = net

            net = tf.nn.lrn(net, 4, bias=1.0,
                            alpha=0.001 / 9.0, beta=0.75, name='norm1')
            net = layers.conv2d(net, 64, [5, 5], scope='conv2')
            end_points['conv2'] = net

            net = tf.nn.lrn(net, 4, bias=1.0,
                            alpha=0.001 / 9.0, beta=0.75, name='norm2')
            net = layers.max_pool2d(net, [2, 2], 2, scope='pool2')
            end_points['pool2'] = net

            net = layers.flatten(net)
            end_points['flatten'] = net

            net = layers.fully_connected(net, 384, scope='fc3')
            end_points['fc3'] = net

            net = layers.dropout(net, dropout_keep_prob,
                                 is_training=is_training, scope='dropout3')
            net = layers.fully_connected(net, 192, scope='fc4')
            end_points['fc4'] = net

            logits = layers.fully_connected(
                net, num_classes,
                biases_initializer=tf.zeros_initializer(),
                weights_initializer=tf.truncated_normal_initializer(
                    stddev=1 / 192.0),
                weights_regularizer=None,
                activation_fn=None,
                scope='logits')

            end_points['logits'] = logits
            end_points['prediction'] = layers.softmax(
                logits, scope='prediction')

        return logits, end_points
