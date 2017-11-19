# -*- coding: utf-8 -*-
""" updated: 2017/6/14
"""
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.layers as layers


def activation_fn(name):
    """ choose different activation function
    """
    if name is 'relu':
        return tf.nn.relu
    elif name is 'sigmoid':
        return tf.nn.sigmoid
    elif name is 'tanh':
        return tf.nn.tanh
    elif name is 'elu':
        return tf.nn.elu
    else:
        return tf.nn.tanh


def initializer_fn(name):
    """
    """
    if name is 'zeros':
        return tf.zeros_initializer
    elif name is 'orthogonal':
        return tf.orthogonal_initializer
    elif name is 'normal':
        return tf.truncated_normal_initializer
    elif name is 'xavier':
        return layers.xavier_initializer
    elif name is 'uniform':
        return tf.random_uniform_initializer
    else:
        raise ValueError('Unknown input type %s' % name)


def rnn_cell(name):
    """ choose different rnn cell
    """
    if name is 'rnn':
        return rnn.BasicRNNCell
    elif name is 'gru':
        return rnn.GRUCell
    elif name is 'basic_lstm':
        return rnn.BasicLSTMCell
    elif name is 'lstm':
        return rnn.LSTMCell
    else:
        raise ValueError('Unknown input type %s' % name)
