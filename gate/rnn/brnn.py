# -*- coding: utf-8 -*-
""" updated: 2017/6/14
    bidirectional rnn model for automatic speech recognition
"""
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from gate.rnn import component


class brnn():
    """
    rnn_cfg.num_layer
    rnn_cfg.activation_fn
    rnn_cfg.rnn_cell
    rnn_cfg.initializer
    rnn_cfg.num_layers
    rnn_cfg.num_units
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
        a. num_layers
        b. timesteps
        c. cell_fn
        d. activation_fn
        e. batch_size
        f. num_units
        """
        # X shape is [batchsize, time_step, feature]
        n_steps = audio_cfg.frames
        n_dim = audio_cfg.frame_length
        batch_size = X.get_shape().as_list()[0]

        # reshape
        X = tf.reshape(X, [-1, n_steps, n_dim])

        # transform to list
        X = tf.unstack(X, n_steps, axis=1)

        # sequence_length
        sequence_length = [n_dim for _ in range(batch_size)]

        # multi-layers
        hidden_input = X
        for idx_layer in range(self.num_layers):
            scope = 'layer_' + str(idx_layer + 1)

            # define
            forward_cell = self.cell_fn(
                self.num_units, activation=self.activation_fn)
            backward_cell = self.cell_fn(
                self.num_units, activation=self.activation_fn)

            # brnn
            # forward-backward, forward final state, backward final state
            fbH, fst, bst = rnn.static_bidirectional_rnn(
                forward_cell, backward_cell, hidden_input, dtype=tf.float32,
                sequence_length=sequence_length, scope=scope)

            fbHrs = [tf.reshape(t, [batch_size, 2, self.num_units])
                     for t in fbH]

            if idx_layer != self.num_layers - 1:
                # output size is [seqlength, batchsize, 2, num_units]
                output = tf.convert_to_tensor(fbHrs, dtype=tf.float32)

                # output size is [seqlength, batchsize, num_units]
                output = tf.reduce_sum(output, 2)

                # from [seqlenth, batchsize, num_units]
                # to [batchsize, seqlenth, num_units]
                hidden_input = tf.unstack(
                    tf.transpose(output, [1, 0, 2]), n_steps, axis=1)

        # sum fw and bw
        # [num_steps, batchsize, n_dim]
        output = tf.convert_to_tensor(fbHrs, dtype=tf.float32)
        output = tf.reduce_sum(fbHrs, axis=2)
        return output
