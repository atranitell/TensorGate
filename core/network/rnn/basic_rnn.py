# -*- coding: utf-8 -*-
""" updated: 2017/6/14
    basic lstm model for automatic speech recognition
"""
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from core.network.rnn.ops import *


def basic_rnn(x, config, is_training):
  """ input args
  """
  # params
  cell_fn = config.cell_fn
  activation_fn = config.activation_fn
  initializer_fn = config.initializer_fn
  dropout_keep = config.dropout_keep
  num_units = config.num_units
  num_layers = config.num_layers

  # get shape and reshape
  batchsize, n_dim, n_step = get_shape(x)
  x = tf.reshape(x, [-1, n_step, n_dim])
  # transform to list
  x = tf.unstack(x, n_step, axis=1)

  # define multi-layer
  cells = []
  for idx in range(num_layers):
    if cell_fn != rnn.LSTMCell:
      rnn_cell = cell_fn(num_units[idx], activation=activation_fn)
    else:
      rnn_cell = cell_fn(num_units[idx], activation=activation_fn,
                         initializer=initializer_fn)
    # define dropout
    if dropout_keep is not None and is_training:
      rnn_cell = rnn.DropoutWrapper(rnn_cell, output_keep_prob=dropout_keep)
    # add in
    cells.append(rnn_cell)

  # define multilayer or else
  stack_cells = rnn.MultiRNNCell(cells, state_is_tuple=True)

  # get lstm cell output
  outputs, _ = rnn.static_rnn(stack_cells, x, dtype=tf.float32)

  logit = inner_product(outputs[-1], 1)
  return logit, outputs
