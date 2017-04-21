"""Provides utilities to preprocess images in AVEC2014 flow.
"""

import tensorflow as tf


def preprocess_for_train(image, output_height, output_width, channels):
    """ Preprocesses the given image for training."""
    # Transform the image to floats.
    image = tf.to_float(image)

    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(
        image, [output_height, output_width, channels])

    # # Randomly flip the image horizontally.
    # distorted_image = tf.image.random_flip_left_right(distorted_image)
    # distorted_image = tf.image.resize_image_with_crop_or_pad(
    #         image, output_height, output_width)

    # Subtract off the mean and divide by the variance of the pixels.
    return tf.image.per_image_standardization(distorted_image)


def preprocess_for_eval(image, output_height, output_width):
    """ Preprocesses the given image for evaluation. """
    # Transform the image to floats.
    image = tf.to_float(image)

    # Resize and crop if needed.
    resized_image = tf.image.resize_image_with_crop_or_pad(
        image, output_height, output_width)

    # Subtract off the mean and divide by the variance of the pixels.
    return tf.image.per_image_standardization(resized_image)


def preprocess_image(image, output_height, output_width, is_training=False, channels=3):
    """ Preprocesses the given image. """
    with tf.name_scope('avec2014_flow'):
        if is_training:
            with tf.name_scope('train'):
                return preprocess_for_train(image, output_height, output_width, channels)
        else:
            with tf.name_scope('test'):
                return preprocess_for_eval(image, output_height, output_width)
