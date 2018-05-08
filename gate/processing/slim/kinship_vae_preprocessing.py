# Copyright 2017 The KaiJIN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides utilities to preprocess images in KINSHIP FOR VAE."""

import tensorflow as tf


def preprocess_for_train(image, output_height, output_width):
  """ Preprocesses the given image for training."""
  # Transform the image to floats.
  image = tf.to_float(image) / 255.
  image = tf.image.resize_images(image, (output_height, output_width))
  return image


def preprocess_for_eval(image, output_height, output_width):
  """ Preprocesses the given image for evaluation. """
  # Transform the image to floats.
  image = tf.to_float(image) / 255.
  image = tf.image.resize_images(image, (output_height, output_width))
  return image


def preprocess_image(image, output_height, output_width, is_training=False):
  """ Preprocesses the given image. """
  if is_training:
    return preprocess_for_train(image, output_height, output_width)
  else:
    return preprocess_for_eval(image, output_height, output_width)
