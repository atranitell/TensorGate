"""Provides utilities to preprocess images in KINSHIP FOR VAE.
    No whiten operation.
"""
import tensorflow as tf


def preprocess_for_train(image, output_height, output_width):
  """ Preprocesses the given image for training."""
  # Transform the image to floats.
  image = tf.to_float(image) / 255.
  image = tf.image.resize_images(image, (output_height, output_width))
  # image = tf.image.random_flip_left_right(image)
  # image = tf.image.random_hue(image, max_delta=0.05)
  # image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
  # image = tf.image.random_brightness(image, max_delta=0.2)
  # image = tf.image.random_saturation(image, lower=0.0, upper=2.0)
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
