# -*- coding: utf-8 -*-
""" Updated: 2017/3/28
"""

import os
import numpy as np
import tensorflow as tf

from core.database import data_entry
from core.database import data_prefetch
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


def load_image_from_text(config):
  """ a normal loader method from text to parse content
  Format:
    path label
    path-to-fold/img0 0
    path-to-fold/img1 10
  """
  # parse
  res, count = data_entry.parse_from_text(
      config.data.entry_path, (str, int), (True, False))

  # construct a fifo queue
  image_list = tf.convert_to_tensor(res[0], dtype=tf.string)
  label_list = tf.convert_to_tensor(res[1], dtype=tf.int32)
  path, label = tf.train.slice_input_producer(
      [image_list, label_list], shuffle=config.data.shuffle)

  image = _load_image(path, config.data.configs[0], config.phase)
  return data_prefetch.generate_batch(image, label, path, config.data)


def load_pair_image_from_text(config):
  """
  Format:
    path label
    path-to-fold/img0 path-to-fold/img0' 0
    path-to-fold/img1 path-to-fold/img1' 10
  """
  # parse
  res, count = data_entry.parse_from_text(
      config.data.entry_path, (str, str, int), (True, True, False))

  # construct a fifo queue
  image_list1 = tf.convert_to_tensor(res[0], dtype=tf.string)
  image_list2 = tf.convert_to_tensor(res[1], dtype=tf.string)
  label_list = tf.convert_to_tensor(res[2], dtype=tf.int32)
  path1, path2, label = tf.train.slice_input_producer(
      [image_list1, image_list2, label_list], shuffle=config.data.shuffle)

  # preprocessing
  image1 = _load_image(path1, config.data.configs[0], config.phase)
  image2 = _load_image(path2, config.data.configs[1], config.phase)

  return data_prefetch.generate_batch(
      [image1, image2], label, [path1, path2], config.data)


def load_triple_image_with_cond(config):
  """
  Format:
    path1 path2 path3 label cond
    path-to-fold/img0 path-to-fold/img0' path-to-fold/img0'' 1 0
    path-to-fold/img1 path-to-fold/img1' path-to-fold/img1'' -1 0
  """
  # parse
  res, count = data_entry.parse_from_text(
      config.data.entry_path,
      (str, str, str, int, int),
      (True, True, True, False, False))

  # construct a fifo queue
  image1 = tf.convert_to_tensor(res[0], dtype=tf.string)
  image2 = tf.convert_to_tensor(res[1], dtype=tf.string)
  image3 = tf.convert_to_tensor(res[2], dtype=tf.string)
  label = tf.convert_to_tensor(res[3], dtype=tf.int32)
  cond = tf.convert_to_tensor(res[4], dtype=tf.int32)
  path1, path2, path3, label, cond = tf.train.slice_input_producer(
      [image1, image2, image3, label, cond], shuffle=config.data.shuffle)

  # preprocessing
  image1 = _load_image(path1, config.data.configs[0], config.phase)
  image2 = _load_image(path2, config.data.configs[1], config.phase)
  image3 = _load_image(path3, config.data.configs[2], config.phase)

  return data_prefetch.generate_batch(
      [image1, image2, image3], [label, cond], [path1, path2, path3], config.data)


def load_npy_from_text(config):
  """
  """
  res, count = data_entry.parse_from_text(
      config.data.entry_path, (str, float), (True, False))
  cfg = config.data.configs[0]

  files = tf.convert_to_tensor(res[0], dtype=tf.string)
  labels = tf.convert_to_tensor(res[1], dtype=tf.float32)
  filename, label = tf.train.slice_input_producer(
      [files, labels], shuffle=config.data.shuffle)

  def load(filepath):
    """ load image from npy file. """
    file_path_abs = str(filepath, encoding='utf-8')
    data = np.load(file_path_abs)
    data = np.float32(np.reshape(data, data.shape))
    return data

  content = tf.py_func(load, [filename], tf.float32)
  content = tf.reshape(content, cfg.shape)

  return data_prefetch.generate_batch(content, label, filename, config.data)


def load_pair_npy_from_text(config):
  """
  """
  res, count = data_entry.parse_from_text(
      config.data.entry_path, (str, str, float), (True, True, False))
  cfg = config.data.configs[0]

  files1 = tf.convert_to_tensor(res[0], dtype=tf.string)
  files2 = tf.convert_to_tensor(res[1], dtype=tf.string)
  labels = tf.convert_to_tensor(res[2], dtype=tf.float32)
  fname1, fname2, label = tf.train.slice_input_producer(
      [files1, files2, labels], shuffle=config.data.shuffle)

  def load(filepath):
    """ load image from npy file. """
    file_path_abs = str(filepath, encoding='utf-8')
    data = np.load(file_path_abs)
    data = np.float32(np.reshape(data, data.shape))
    return data

  content1 = tf.py_func(load, [fname1], tf.float32)
  content1 = tf.reshape(content1, cfg.shape)

  content2 = tf.py_func(load, [fname2], tf.float32)
  content2 = tf.reshape(content2, cfg.shape)

  return data_prefetch.generate_batch(
      [content1, content2], label, [fname1, fname2], config.data)
