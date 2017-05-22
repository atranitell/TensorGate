"""Provides utilities to preprocess images in MNIST.
"""

import tensorflow as tf


def preprocess_for_train(image, output_height, output_width):
    """ Preprocesses the given image for training."""
    # Transform the image to floats.
    image = tf.to_float(image)

    # Randomly crop a [height, width] section of the image.
    resized_image = tf.image.resize_images(
        image, (output_height, output_width))
    # tf.summary.image('image_crop', tf.expand_dims(distorted_image, 0))

    # Subtract off the mean and divide by the variance of the pixels.
    return tf.image.per_image_standardization(resized_image)


def preprocess_for_eval(image, output_height, output_width):
    """ Preprocesses the given image for evaluation. """
    # Transform the image to floats.
    image = tf.to_float(image)

    # Resize and crop if needed.
    resized_image = tf.image.resize_image_with_crop_or_pad(
        image, output_width, output_height)

    # Subtract off the mean and divide by the variance of the pixels.
    return tf.image.per_image_standardization(resized_image)


def preprocess_image(image, output_height, output_width, is_training=False):
    """ Preprocesses the given image. """
    if is_training:
        return preprocess_for_train(image, output_height, output_width)
    else:
        return preprocess_for_eval(image, output_height, output_width)
