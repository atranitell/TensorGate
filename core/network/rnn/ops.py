# -*- coding: utf-8 -*-
""" Common func for rnns
"""
from tensorflow.contrib import layers


def get_shape(x):
  """ used for (batchsize, n_dim, n_step)
  """
  x_shape = x.get_shape().as_list()
  if len(x_shape) == 4:
    # for NHWC data
    batchsize = x_shape[0]
    n_dim = x_shape[1] * x_shape[2]
    n_step = x_shape[3]
  elif len(x_shape) == 3:
    # for N,Sequence,C
    batchsize = x_shape[0]
    n_dim = x_shape[1]
    n_step = x_shape[2]
  else:
    raise ValueError('Unknown input shape.')
  return batchsize, n_dim, n_step


def inner_product(x, num_output, scope="linear"):
  return layers.fully_connected(
      inputs=x,
      num_outputs=num_output,
      activation_fn=None,
      weights_initializer=layers.xavier_initializer(),
      scope=scope)
