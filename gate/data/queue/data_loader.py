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

import tensorflow as tf
from gate.data.queue import data_entry
from gate.data.queue import data_prefetch
from gate.data.queue import data_utils
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
