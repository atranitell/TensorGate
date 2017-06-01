"""Provides utilities to preprocess images in KinFace.
"""
import tensorflow as tf


def preprocess_for_train(image, output_height,
                         output_width, channels):
    """ Preprocesses the given image for training."""
    # tf.summary.image('image_raw', tf.expand_dims(image, 0))
    image = tf.to_float(image)

    # resize to [160, 160] and then add padding to [184, 184]
    image = tf.image.resize_images(
        image, [output_height, output_width],
        method=tf.image.ResizeMethod.AREA)
    image = tf.image.resize_image_with_crop_or_pad(image, 184, 184)

    image = tf.random_crop(
        image, [output_width, output_height, channels])
    tf.summary.image('image_crop', tf.expand_dims(image, 0))

    image = tf.image.random_flip_left_right(image)
    tf.summary.image('image_flip', tf.expand_dims(image, 0))

    # distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    # tf.summary.image('image_brightness', tf.expand_dims(distorted_image, 0))

    # distorted_image = tf.image.random_contrast(
    #     distorted_image, lower=0.2, upper=1.8)
    # tf.summary.image('image_contrast', tf.expand_dims(distorted_image, 0))

    return image


def preprocess_for_eval(image, output_height, output_width):
    """ Preprocesses the given image for evaluation. """
    # tf.summary.image('image_raw', tf.expand_dims(image, 0))
    image = tf.to_float(image)
    image = tf.image.resize_images(
        image, [output_height, output_width],
        method=tf.image.ResizeMethod.AREA)
    return image


def preprocess_image(image, output_height, output_width,
                     is_training=False, channels=3):
    """ Preprocesses the given image. """
    if is_training:
        return preprocess_for_train(image, output_height, output_width, channels)
    else:
        return preprocess_for_eval(image, output_height, output_width)
