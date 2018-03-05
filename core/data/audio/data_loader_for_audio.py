# -*- coding: utf-8 -*-
""" Updated: 2017/3/28
"""
import random
import numpy as np
import tensorflow as tf
from core.data import data_entry
from core.data import data_prefetch
from core.data import data_utils


def _load_audio(filepath, start_idx, fnum, flen, finvl):
  file_path_abs = str(filepath, encoding='utf-8')
  data = np.load(file_path_abs)

  valid_length = data.shape[0] - fnum * flen - finvl
  if start_idx < 0:
    start = random.randint(0, valid_length)
  else:
    start = start_idx

  audio_data = np.array([])
  for i in range(fnum):
    start_i = start + i * finvl
    _data = data[start_i: start_i + flen]
    if start_idx < 0:
      _data += np.random.normal(0.001, 1.0)
    audio_data = np.append(audio_data, _data)

  audio_data = np.float32(np.reshape(audio_data, [fnum, flen]))
  return audio_data


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
      inp=[tf_input[0], tf_input[1], cfg.frame_num,
           cfg.frame_length, cfg.frame_invl],
      Tout=tf.float32)

  files = tf.reshape(files, [cfg.frame_num, cfg.frame_length])
  paths = tf_input[0]
  extras = tf_input[2]

  return data_prefetch.generate_batch(files, extras, paths, config.data)
