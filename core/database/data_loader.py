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
      [image1, image2], label, path1, config.data)
