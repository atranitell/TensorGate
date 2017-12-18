import tensorflow as tf
from tensorflow.contrib import layers
from core.utils.logger import logger


def bn(x, is_training):
  return layers.batch_norm(
      inputs=x,
      decay=0.9,
      updates_collections=None,
      epsilon=1e-5,
      scale=True,
      is_training=is_training)


def conv2d(x, filters, ksize, stride, name="conv2d"):
  return tf.layers.conv2d(
      inputs=x,
      filters=filters,
      kernel_size=ksize,
      strides=stride,
      padding='VALID',
      use_bias=False,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
      kernel_regularizer=layers.l2_regularizer(0.0005),
      name=name)


def lrelu(x, leak=0.2):
  return tf.maximum(x, leak * x)


def fc(x, filters, name):
  return layers.fully_connected(
      x, filters,
      biases_initializer=None,
      weights_initializer=layers.xavier_initializer(),
      weights_regularizer=None,
      activation_fn=None,
      scope=name)


def simplenet(x, num_classes, is_training, reuse=None):
  """
  """
  with tf.variable_scope('simplenet_bn_sign', reuse=reuse):
    end_points = {}
    x = tf.sign(x)
    net = lrelu(bn(conv2d(x, 32, (7, 7), (2, 2), name='conv1'), is_training))
    net = lrelu(bn(conv2d(net, 64, (5, 5), (1, 1), name='conv2'), is_training))
    net = lrelu(bn(conv2d(net, 64, (5, 5), (2, 2), name='conv3'), is_training))
    net = lrelu(bn(conv2d(net, 128, (4, 4), (1, 1), name='conv4'), is_training))
    net = lrelu(bn(conv2d(net, 128, (4, 4), (2, 2), name='conv5'), is_training))
    net = conv2d(net, 256, (4, 4), (2, 2), name='conv6')
    net = layers.dropout(net, keep_prob=0.8, is_training=is_training)
    net = tf.reduce_mean(net, [1, 2])
    logits = fc(net, num_classes, 'logits')
    return logits, end_points
