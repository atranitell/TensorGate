# -*- coding: utf-8 -*-
""" Updated: 2017/3/28
"""

import os
import numpy as np
import tensorflow as tf

from core.database import data_entry
from core.database import data_prefetch
from core.database.preprocessing.factory import preprocessing


def load_image_from_text(cfg, phase):
  """ a normal loader method from text to parse content
  Format:
    path label
    path-to-fold/img0 0
    path-to-fold/img1 10
  """
  # parse
  res, count = data_entry.parse_from_text(
      cfg['entry_path'], (str, int), (True, False))
  cfg['total_num'] = count

  image_list = res[0]
  label_list = res[1]

  # construct a fifo queue
  image_list = tf.convert_to_tensor(image_list, dtype=tf.string)
  label_list = tf.convert_to_tensor(label_list, dtype=tf.int32)
  path, label = tf.train.slice_input_producer(
      [image_list, label_list], shuffle=cfg['shuffle'])

  # preprocessing
  image_raw = tf.read_file(path)
  image = tf.image.decode_image(image_raw, channels=cfg['image']['channels'])
  image = tf.reshape(image, [cfg['image']['raw_height'],
                             cfg['image']['raw_width'],
                             cfg['image']['channels']])
  process_fn = preprocessing(cfg['image']['preprocessing_method'])
  image = process_fn(image, phase, cfg['image'])

  return data_prefetch.generate_batch(image, label, path, cfg)
