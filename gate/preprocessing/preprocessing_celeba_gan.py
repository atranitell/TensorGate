"""Provides utilities to preprocess images FOR GAN.
"""
import tensorflow as tf


def preprocess_for_train(image, output_height, output_width):
    """ Preprocesses the given image for training."""
    # Transform the image to floats.
    image = tf.to_float(image) / 255.0
    resized_image = tf.image.resize_image_with_crop_or_pad(
        image, output_width, output_height)
    return resized_image


def preprocess_for_eval(image, output_height, output_width):
    """ Preprocesses the given image for evaluation. """
    # Transform the image to floats.
    image = tf.to_float(image) / 255.0
    resized_image = tf.image.resize_image_with_crop_or_pad(
        image, output_width, output_height)
    return resized_image


def preprocess_image(image, output_height, output_width, is_training=False):
    """ Preprocesses the given image. """
    with tf.name_scope('celeba_gan'):
        if is_training:
            with tf.name_scope('train'):
                return preprocess_for_train(image, output_height, output_width)
        else:
            with tf.name_scope('test'):
                return preprocess_for_eval(image, output_height, output_width)
