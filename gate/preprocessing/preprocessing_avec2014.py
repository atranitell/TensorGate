"""Provides utilities to preprocess images in AVEC2014.
"""

import tensorflow as tf


def preprocess_for_train(image, output_height, output_width, channels):
    """ Preprocesses the given image for training."""
    # tf.summary.image('image_raw', tf.expand_dims(image, 0))

    # Transform the image to floats.
    image = tf.to_float(image)

    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(image, [output_height, output_width, channels])
    # tf.summary.image('image_crop', tf.expand_dims(distorted_image, 0))

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    # tf.summary.image('image_flip', tf.expand_dims(distorted_image, 0))

    # Because these operations are not commutative, consider randomizing
    # the order their operation.

    # distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    # tf.summary.image('image_brightness', tf.expand_dims(distorted_image, 0))

    # distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    # tf.summary.image('image_contrast', tf.expand_dims(distorted_image, 0))

    # Subtract off the mean and divide by the variance of the pixels.
    return tf.image.per_image_standardization(distorted_image)


def preprocess_for_eval(image, output_height, output_width):
    """ Preprocesses the given image for evaluation. """
    # tf.summary.image('image_raw', tf.expand_dims(image, 0))

    # Transform the image to floats.
    image = tf.to_float(image)

    # Resize and crop if needed.
    resized_image = tf.image.resize_image_with_crop_or_pad(image, output_width, output_height)
    # tf.summary.image('image_resize', tf.expand_dims(resized_image, 0))

    # Subtract off the mean and divide by the variance of the pixels.
    return tf.image.per_image_standardization(resized_image)


def preprocess_image(image, output_height, output_width, is_training=False, channels=3):
    """ Preprocesses the given image. """
    with tf.name_scope('avec2014'):
        if is_training:
            with tf.name_scope('train'):
                return preprocess_for_train(image, output_height, output_width, channels)
        else:
            with tf.name_scope('test'):
                return preprocess_for_eval(image, output_height, output_width)
