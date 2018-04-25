# -*- coding: utf-8 -*-
"""
GATE FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2017/8/28

--------------------------------------------------------

Package into batch to feed to session.

"""

import tensorflow as tf


def generate_batch(X, Y, Z, cfg):
  """ For single X, Y and Z.
  """
  if cfg.shuffle:
    Xs, Ys, Zs = tf.train.shuffle_batch(
        tensors=[X, Y, Z],
        batch_size=cfg.batchsize,
        capacity=cfg.min_queue_num + 3 * cfg.batchsize,
        min_after_dequeue=cfg.min_queue_num,
        num_threads=cfg.reader_thread)
  else:
    Xs, Ys, Zs = tf.train.batch(
        tensors=[X, Y, Z],
        batch_size=cfg.batchsize,
        capacity=cfg.min_queue_num + 3 * cfg.batchsize,
        num_threads=cfg.reader_thread)

  return Xs, Ys, Zs
