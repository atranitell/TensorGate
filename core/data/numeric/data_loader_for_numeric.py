# -*- coding: utf-8 -*-
""" Updated: 2017/3/28
"""
import numpy as np
import tensorflow as tf
from core.data import data_entry
from core.data import data_prefetch
from core.data import data_utils


def _load_npy(filepath):
  file_path_abs = str(filepath, encoding='utf-8')
  data = np.load(file_path_abs)
  data = np.float32(np.reshape(data, data.shape))
  return data


def load_npy(config):
  """ Any format
  """
  data = config.data
  res, _ = data_entry.parse_from_text(
      data.entry_path, data.entry_dtype, data.entry_check)
  cfg = data.configs[0]

  tf_inputs = data_utils.convert_to_tensor(res, data.entry_dtype)
  tf_input = tf.train.slice_input_producer(tf_inputs, shuffle=data.shuffle)

  files, paths, extras = [], [], []
  for i, t in enumerate(data.entry_check):
    if t:
      content = tf.py_func(_load_npy, [tf_input[i]], tf.float32)
      content = tf.reshape(content, cfg.shape)
      files.append(content)
      paths.append(tf_input[i])
    else:
      extras.append(tf_input[i])

  files = files[0] if len(files) == 1 else files
  paths = paths[0] if len(paths) == 1 else paths
  extras = extras[0] if len(extras) == 1 else extras

  return data_prefetch.generate_batch(files, extras, paths, config.data)
