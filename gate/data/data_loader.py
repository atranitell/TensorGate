# -*- coding: utf-8 -*-
""" updated: 2017/3/28
"""

import tensorflow as tf
from gate.data import data_entry
from gate.data import data_loader_for_video


def load_single_jpg_from_text(data_path, shuffle=True):
    """ decode jpg images from jpg-file-path
    """
    image_list, label_list, _ = data_entry.read_image_from_text_list_with_label(data_path)

    # construct a fifo queue
    # images = tf.convert_to_tensor(image_list, dtype=tf.string)
    # labels = tf.convert_to_tensor(label_list, dtype=tf.int32)
    imgpath, label = tf.train.slice_input_producer([image_list, label_list], shuffle=shuffle)

    # preprocessing
    image_raw = tf.read_file(imgpath)
    image_jpeg = tf.image.decode_jpeg(image_raw, channels=3)

    # image, label, filename
    return image_jpeg, label, imgpath


def load_single_video_frame_from_text(data_path, channels=16, is_training=True, shuffle=True):
    """ load video sequence from a folder
    e.g.
        a. |12345|12345|12345|12345|12345|12345|...
        b. | 2   |1    |   4 |  3  |1    |  3  |...
        c. 214313 = 6*3channels = 18 channels

    args:
        channels: how much images in a folder will be compress into a image.
    """
    folds, labels = data_entry.read_fold_from_text_list_with_label(data_path)

    # construct a fifo queue
    foldname, label = tf.train.slice_input_producer([folds, labels], shuffle=shuffle)

    combined_image = tf.py_func(data_loader_for_video.compress_multi_imgs_to_one,
                                [foldname, channels, is_training], tf.uint8)

    return combined_image, label, foldname


def load_pair_video_frame_from_text(data_path, channels=16, is_training=True, shuffle=True):
    """ load pair video sequence from a folder
    """
    fold_0, fold_1, labels = data_entry.read_pair_folds_from_text_list_with_label(data_path)

    # construct a fifo queue
    fold_0_path, fold_1_path, label = tf.train.slice_input_producer(
        [fold_0, fold_1, labels], shuffle=shuffle)

    img1, img2 = tf.py_func(data_loader_for_video.compress_pair_multi_imgs_to_one,
                            [fold_0_path, fold_1_path, channels, is_training],
                            [tf.uint8, tf.uint8])

    return img1, img2, label, fold_0_path, fold_1_path
