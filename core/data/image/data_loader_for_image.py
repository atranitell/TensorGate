# -*- coding: utf-8 -*-
""" Updated: 2017/3/28
"""
import tensorflow as tf
from core.data import data_entry
from core.data import data_prefetch
from core.data import data_utils
from core.data.image.preprocessing.factory import preprocessing


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
  image = preprocessing(X=image,
                        name=cfgimg.preprocessing_method,
                        cfg=cfgimg,
                        phase=phase)
  return image


def load_image(config):
  """ Any format
  """
  data = config.data
  res, _ = data_entry.parse_from_text(
      data.entry_path, data.entry_dtype, data.entry_check)

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
