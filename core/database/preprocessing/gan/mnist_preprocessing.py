"""Provides utilities to preprocess images in MNIST FOR GAN.
    No whiten operation.
"""
import tensorflow as tf


def preprocess_for_train(image, output_height, output_width):
  """ Preprocesses the given image for training."""
  # Transform the image to floats.
  image = tf.to_float(image) / 255.
  resized_image = tf.image.resize_images(
      image, (output_height, output_width))
  return resized_image


def preprocess_for_eval(image, output_height, output_width):
  """ Preprocesses the given image for evaluation. """
  # Transform the image to floats.
  image = tf.to_float(image) / 255.
  resized_image = tf.image.resize_images(
      image, (output_height, output_width))
  return resized_image


def preprocess_image(image, output_height, output_width, is_training=False):
  """ Preprocesses the given image. """
  if is_training:
    return preprocess_for_train(image, output_height, output_width)
  else:
    return preprocess_for_eval(image, output_height, output_width)
