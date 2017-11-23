# -*- coding: utf-8 -*-
""" Updated: 2017/3/28
"""

import os
import numpy as np
import tensorflow as tf

from core.database import data_entry
from core.database import data_prefetch
from core.database.preprocessing.factory import preprocessing


def load_image_from_text(config):
  """ a normal loader method from text to parse content
  Format:
    path label
    path-to-fold/img0 0
    path-to-fold/img1 10
  """
  # setting
  phase = config.phase
  cfgdata = config.data
  cfgimg = config.data.configs[0]

  # parse
  res, count = data_entry.parse_from_text(
      cfgdata.entry_path, (str, int), (True, False))

  image_list = res[0]
  label_list = res[1]

  # construct a fifo queue
  image_list = tf.convert_to_tensor(image_list, dtype=tf.string)
  label_list = tf.convert_to_tensor(label_list, dtype=tf.int32)
  path, label = tf.train.slice_input_producer(
      [image_list, label_list], shuffle=cfgdata.shuffle)

  # preprocessing
  image_raw = tf.read_file(path)
  image = tf.image.decode_image(image_raw, channels=cfgimg.channels)
  image = tf.reshape(image, [cfgimg.raw_height,
                             cfgimg.raw_width,
                             cfgimg.channels])
  
  process_fn = preprocessing(cfgimg.preprocessing_method)
  image = process_fn(image, phase, cfgimg)

  return data_prefetch.generate_batch(image, label, path, cfgdata)


def load_pair_image_from_text(cfg, phase):
  """
  Format:
    path label
    path-to-fold/img0 path-to-fold/img0' 0
    path-to-fold/img1 path-to-fold/img1' 10
  """
  # parse
  res, count = data_entry.parse_from_text(
      cfg['entry_path'], (str, str, int), (True, True, False))
  cfg['total_num'] = count

  image_list1 = res[0]
  image_list2 = res[1]
  label_list = res[2]

  # construct a fifo queue
  image_list1 = tf.convert_to_tensor(image_list1, dtype=tf.string)
  image_list2 = tf.convert_to_tensor(image_list2, dtype=tf.string)
  label_list = tf.convert_to_tensor(label_list, dtype=tf.int32)
  path1, path2, label = tf.train.slice_input_producer(
      [image_list1, image_list2, label_list], shuffle=cfg['shuffle'])

  # preprocessing
  image_raw1 = tf.read_file(path1)
  image1 = tf.image.decode_image(image_raw1, channels=cfg['image']['channels'])
  image1 = tf.reshape(image1, [cfg['image']['raw_height'],
                             cfg['image']['raw_width'],
                             cfg['image']['channels']])
  process_fn = preprocessing(cfg['image']['preprocessing_method'])
  image1 = process_fn(image1, phase, cfg['image'])

  image_raw2 = tf.read_file(path2)
  image2 = tf.image.decode_image(image_raw2, channels=cfg['image']['channels'])
  image2 = tf.reshape(image2, [cfg['image']['raw_height'],
                             cfg['image']['raw_width'],
                             cfg['image']['channels']])
  process_fn = preprocessing(cfg['image']['preprocessing_method'])
  image2 = process_fn(image2, phase, cfg['image'])

  return data_prefetch.generate_batch([image1, image2], label, path1, cfg)
