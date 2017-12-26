# -*- coding: utf-8 -*-
""" Updated: 2017/3/28
"""
import random
import numpy as np
import tensorflow as tf
from core.database import data_entry
from core.database.queue import data_prefetch
from core.database.preprocessing.factory import preprocessing


def _load_image(path, cfgimg, phase):
  """ from path load and set image.
  """
  # load in
  image_raw = tf.read_file(path)
  image = tf.image.decode_image(image_raw, channels=cfgimg.channels)
  image = tf.reshape(image, [cfgimg.raw_height,
                             cfgimg.raw_width,
                             cfgimg.channels])

  # if graying
  if cfgimg.gray:
    image = tf.image.rgb_to_grayscale(image)

  # preprocessing
  process_fn = preprocessing(cfgimg.preprocessing_method)
  image = process_fn(image, phase, cfgimg)
  return image


def _convert_to_tensor(res_list, type_list):
  tensors = []
  for i, t in enumerate(type_list):
    if t == str:
      r = tf.convert_to_tensor(res_list[i], dtype=tf.string)
    elif t == int:
      r = tf.convert_to_tensor(res_list[i], dtype=tf.int32)
    elif t == float:
      r = tf.convert_to_tensor(res_list[i], dtype=tf.float32)
    else:
      raise ValueError('Unknown Input Type [%s]' % t)
    tensors.append(r)
  return tensors


def load_image(config):
  """ Any format
  """
  data = config.data
  res, _ = data_entry.parse_from_text(
      data.entry_path, data.entry_dtype, data.entry_check)

  tf_inputs = _convert_to_tensor(res, data.entry_dtype)
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


def load_npy(config):
  """ Any format
  """
  data = config.data
  res, _ = data_entry.parse_from_text(
      data.entry_path, data.entry_dtype, data.entry_check)
  cfg = data.configs[0]

  tf_inputs = _convert_to_tensor(res, data.entry_dtype)
  tf_input = tf.train.slice_input_producer(tf_inputs, shuffle=data.shuffle)

  def load(filepath):
    file_path_abs = str(filepath, encoding='utf-8')
    data = np.load(file_path_abs)
    data = np.float32(np.reshape(data, data.shape))
    return data

  files, paths, extras = [], [], []
  for i, t in enumerate(data.entry_check):
    if t:
      content = tf.py_func(load, [tf_input[i]], tf.float32)
      content = tf.reshape(content, cfg.shape)
      files.append(content)
      paths.append(tf_input[i])
    else:
      extras.append(tf_input[i])

  files = files[0] if len(files) == 1 else files
  paths = paths[0] if len(paths) == 1 else paths
  extras = extras[0] if len(extras) == 1 else extras

  return data_prefetch.generate_batch(files, extras, paths, config.data)
