# -*- coding: utf-8 -*-
"""
GATE FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2017/3/28

--------------------------------------------------------

Data Loader

"""

import random
import numpy as np
import tensorflow as tf
from gate.data import data_utils
from gate.data.queue import data_entry
from gate.data.queue import data_prefetch
from gate.data.preprocessing.factory import get_preprocessing


def _load_image(path, cfgimg, phase):
  """ from path load and set image.
  """
  # load in
  image_raw = tf.read_file(path)
  image = tf.image.decode_image(image_raw, channels=cfgimg.channels)

  # reshape
  if cfgimg.raw_height > 0 and cfgimg.raw_width > 0:
    image = tf.reshape(image, [cfgimg.raw_height,
                               cfgimg.raw_width,
                               cfgimg.channels])

  # if graying
  if cfgimg.gray:
    image = tf.image.rgb_to_grayscale(image)

  # preprocessing
  image = get_preprocessing(X=image,
                            name=cfgimg.preprocessing_method,
                            cfg=cfgimg,
                            phase=phase)
  return image


def load_image(config):
  """ Any format
  """
  data = config.data
  res, count = data_entry.parse_from_text(
      data.entry_path, data.entry_dtype, data.entry_check)
  # the total num will be written.
  config.data.total_num = count

  tf_inputs = data_utils.convert_to_tensor(res, data.entry_dtype)
  tf_input = tf.train.slice_input_producer(tf_inputs, shuffle=data.shuffle)

  images, paths, extras = [], [], []
  for i, t in enumerate(data.entry_check):
    if t:
      image = _load_image(tf_input[i], data.configs[i], config.phase)
      images.append(image)
      paths.append(tf_input[i])
    else:
      extras.append(tf_input[i])

  images = images[0] if len(images) == 1 else images
  paths = paths[0] if len(paths) == 1 else paths
  extras = extras[0] if len(extras) == 1 else extras

  return data_prefetch.generate_batch(images, extras, paths, data)


def _load_npy(filepath):
  file_path_abs = str(filepath, encoding='utf-8')
  data = np.load(file_path_abs)
  data = np.float32(np.reshape(data, data.shape))
  return data


def load_npy(config):
  """ Any format
  """
  data = config.data
  res, count = data_entry.parse_from_text(
      data.entry_path, data.entry_dtype, data.entry_check)
 # the total num will be written.
  config.data.total_num = count
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
  res, count = data_entry.parse_from_text(
      data.entry_path, data.entry_dtype, data.entry_check)
 # the total num will be written.
  config.data.total_num = count
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
