# -*- coding: utf-8 -*-
""" updated: 2017/6/14
    basic lstm model for automatic speech recognition
"""
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from gate.rnn import component


class basic_rnn():
    """ basic rnn
        user-defined,
            activation function,
            cell_fn
    """

    def __init__(self, rnn_cfg):
        """
        """
        self.activation_fn = component.activation_fn(rnn_cfg.activation)
        self.cell_fn = component.rnn_cell(rnn_cfg.cell)
        self.initializer_fn = component.initializer_fn(rnn_cfg.initializer)
        self.dropout = rnn_cfg.dropout

        self.num_units = rnn_cfg.num_units
        self.num_layers = rnn_cfg.num_layers

    def model(self, X, audio_cfg):
        """ input args
        """
        # X shape is [batchsize, time_step, feature]
        n_steps = audio_cfg.frames
        n_dim = audio_cfg.frame_length

        # reshape
        X = tf.reshape(X, [-1, n_steps, n_dim])

        # transform to list
        X = tf.unstack(X, n_steps, axis=1)

        # define each lstm cell
        if self.cell_fn != rnn.LSTMCell:
            rnn_cell = self.cell_fn(
                self.num_units, activation=self.activation_fn)
        else:
            rnn_cell = self.cell_fn(
                self.num_units, activation=self.activation_fn,
                initializer=self.initializer_fn)

        # define dropout
        if self.dropout is not None:
            rnn_cell = rnn.DropoutWrapper(
                rnn_cell, output_keep_prob=self.dropout)

        # define multilayer or else
        rnn_cell = rnn.MultiRNNCell([rnn_cell] * self.num_layers)

        # get lstm cell output
        outputs, _ = rnn.static_rnn(rnn_cell, X, dtype=tf.float32)

        # print(outputs[-1])

        return outputs
