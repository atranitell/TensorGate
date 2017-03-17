"""Creates a variant of the LeNet model.

Note that since the output is a set of 'logits', the values fall in the
interval of (-infinity, infinity). Consequently, to convert the outputs to a
probability distribution over the characters, one will need to convert them
using the softmax function:

        logits = lenet.lenet(images, is_training=False)
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
from tensorflow.contrib import metrics
from nets import net


class lenet(net.Net):

    def __init__(self):
        self.weight_decay = 0.0005
        self.dropout_keep_prob = 0.9

    def arg_scope(self):
        weight_decay = self.weight_decay

        with arg_scope([layers.conv2d, layers.fully_connected],
                       weights_regularizer=layers.l2_regularizer(weight_decay),
                       weights_initializer=tf.truncated_normal_initializer(
                           stddev=0.1),
                       activation_fn=tf.nn.relu) as sc:
            return sc

    def model(self, images, num_classes, is_training):
        end_points = {}
        dropout_keep_prob = self.dropout_keep_prob

        with tf.variable_scope('lenet', 'LeNet', [images, num_classes]):
            net = layers.conv2d(images, 32, [5, 5], scope='conv1')
            net = layers.max_pool2d(net, [2, 2], 2, scope='pool1')
            net = layers.conv2d(net, 64, [5, 5], scope='conv2')
            net = layers.max_pool2d(net, [2, 2], 2, scope='pool2')
            net = layers.flatten(net)
            end_points['Flatten'] = net

            net = layers.fully_connected(net, 1024, scope='fc3')
            end_points['fc3'] = net
            net = layers.dropout(net, dropout_keep_prob, is_training=is_training,
                                 scope='dropout3')
            logits = layers.fully_connected(net, num_classes, activation_fn=None,
                                            scope='fc4')

        end_points['Logits'] = logits
        end_points['Predictions'] = layers.softmax(logits, scope='Predictions')

        return logits, end_points
