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
        rnn_cell = self.cell_fn(
            self.num_units, activation=self.activation_fn)

        # get lstm cell output
        outputs, _ = rnn.static_rnn(rnn_cell, X, dtype=tf.float32)

        return outputs
