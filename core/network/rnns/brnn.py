# -*- coding: utf-8 -*-
""" updated: 2017/6/14
    bidirectional rnn model for automatic speech recognition
"""
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from core.network.rnns.ops import *


def brnn(x, config, is_training):
  """ input args
  a. num_layers
  b. timesteps
  c. cell_fn
  d. activation_fn
  e. batch_size
  f. num_units
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
  # sequence_length
  sequence_length = [n_dim for _ in range(batchsize)]

  fw_cells = []
  bw_cells = []
  for idx in range(num_layers):
    # define
    fw_cell = cell_fn(num_units[idx], activation=activation_fn)
    bw_cell = cell_fn(num_units[idx], activation=activation_fn)
    # define dropout
    if dropout_keep is not None and is_training:
      fw_cell = rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout_keep)
      bw_cell = rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout_keep)
    fw_cells.append(fw_cell)
    bw_cells.append(bw_cell)

  # stacks multi-layer
  fw_cells = rnn.MultiRNNCell(fw_cells, state_is_tuple=True)
  bw_cells = rnn.MultiRNNCell(bw_cells, state_is_tuple=True)

  # output
  outputs, _, _ = rnn.static_bidirectional_rnn(
      fw_cells, bw_cells, x, dtype=tf.float32, sequence_length=sequence_length)

  logit = inner_product(outputs[-1], 1)
  return logit, outputs
