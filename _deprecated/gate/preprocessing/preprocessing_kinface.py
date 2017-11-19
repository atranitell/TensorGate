"""Provides utilities to preprocess images in KinFace.
"""
import tensorflow as tf


def preprocess_for_train(image, output_height,
                         output_width, channels):
    """ Preprocesses the given image for training."""

    # ---------------------------------------------------
    # # FOR FINETUNE ON INCEPTION-RESNET-V1
    # image = tf.to_float(image)
    # # resize to [160, 160] and then add padding to [184, 184]
    # image = tf.image.resize_images(
    #     image, [output_height, output_width],
    #     method=tf.image.ResizeMethod.AREA)
    # image = tf.image.resize_image_with_crop_or_pad(image, 184, 184)
    # ---------------------------------------------------
    image = tf.to_float(image)
    image = tf.random_crop(
        image, [output_height, output_width, channels])
    image = tf.image.random_flip_left_right(image)

    return tf.image.per_image_standardization(image)  # image


def preprocess_for_eval(image, output_height, output_width):
    """ Preprocesses the given image for evaluation. """
    image = tf.to_float(image)
    # image = tf.image.resize_images(
    #     image, [output_height, output_width],
    #     method=tf.image.ResizeMethod.AREA)
    image = tf.image.resize_image_with_crop_or_pad(
        image, output_width, output_height)

    return tf.image.per_image_standardization(image)  # image


def preprocess_image(image, output_height, output_width,
                     is_training=False, channels=3):
    """ Preprocesses the given image. """
    if is_training:
        return preprocess_for_train(image, output_height, output_width, channels)
    else:
        return preprocess_for_eval(image, output_height, output_width)
