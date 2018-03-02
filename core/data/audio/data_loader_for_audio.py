# -*- coding: utf-8 -*-
""" Updated: 2017/3/28
"""
import random
import numpy as np
import tensorflow as tf
from core.data import data_entry
from core.data import data_prefetch
from core.data import data_utils


def _load_audio(filepath, start_idx, length):
  file_path_abs = str(filepath, encoding='utf-8')
  data = np.load(file_path_abs)
  data = data[start_idx:start_idx + length]
  data = np.float32(np.reshape(data, data.shape))
  return data


def load_audio(config):
  """ the fixed format:
    path_to_audio start_idx label
  """
  data = config.data
  res, _ = data_entry.parse_from_text(
      data.entry_path, data.entry_dtype, data.entry_check)
  cfg = data.configs[0]

  tf_inputs = data_utils.convert_to_tensor(res, data.entry_dtype)
  tf_input = tf.train.slice_input_producer(tf_inputs, shuffle=data.shuffle)

  files = tf.py_func(
      func=_load_audio,
      inp=[tf_input[0], tf_input[1], cfg.length],
      Tout=tf.float32)

  files = tf.reshape(files, [cfg.length, 1])
  paths = tf_input[0]
  extras = tf_input[2]

  return data_prefetch.generate_batch(files, extras, paths, config.data)
