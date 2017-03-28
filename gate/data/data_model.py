# -*- coding: utf-8 -*-
""" updated: 2017/3/28
"""

import tensorflow as tf
from gate import preprocessing

from gate.data import data_loader
from gate.data import data_prefetch


def loads(method, data_path, shuffle, data_type, channels=3,
          preprocessing_method1=None, preprocessing_method2=None,
          raw_height=256, raw_width=256,
          output_height=224, output_width=224,
          min_queue_num=128, batch_size=32, reader_thread=16):
    """ interface
    """
    if method == 'single_image_from_text':
        return load_single_image(
            data_path, shuffle, data_type,
            preprocessing_method1, output_height, output_width,
            min_queue_num, batch_size, reader_thread)

    elif method == 'single_video_from_text':
        return load_single_video_frame(
            data_path, shuffle, data_type, channels, preprocessing_method1,
            raw_height, raw_width, output_height, output_width,
            min_queue_num, batch_size, reader_thread)

    elif method == 'pair_video_from_text':
        return load_pair_video_frame(
            data_path, shuffle, data_type, channels,
            preprocessing_method1, preprocessing_method2,
            raw_height, raw_width, output_height, output_width,
            min_queue_num, batch_size, reader_thread)

    else:
        raise ValueError('Unkonwn method %s' % method)


def load_single_image(data_path, shuffle, data_type,
                      preprocessing_method, output_height, output_width,
                      min_queue_num, batch_size, reader_thread):
    """ load single image
    """
    image, label, filename = data_loader.load_single_jpg_from_text(
        data_path, shuffle=shuffle)

    image = preprocessing.factory.get_preprocessing(
        preprocessing_method, data_type, image,
        output_height, output_width)

    return data_prefetch.generate_img_label_batch(
        image, label, filename, shuffle,
        batch_size, min_queue_num, reader_thread)


def load_single_video_frame(data_path, shuffle, data_type, channels, preprocessing_method,
                            raw_height, raw_width, output_height, output_width,
                            min_queue_num, batch_size, reader_thread):
    """ load single video frame
    """
    is_training = True if data_type is 'train' else False

    image, label, filename = data_loader.load_single_video_frame_from_text(
        data_path, channels, is_training, shuffle=shuffle)

    image = tf.reshape(image, shape=[raw_height, raw_width, channels * 3])

    image = preprocessing.factory.get_preprocessing(
        preprocessing_method, data_type, image,
        output_height, output_width, channels=channels * 3)

    return data_prefetch.generate_img_label_batch(
        image, label, filename, shuffle,
        batch_size, min_queue_num, reader_thread)


def load_pair_video_frame(data_path, shuffle, data_type, channels,
                          preprocessing_method1, preprocessing_method2,
                          raw_height, raw_width, output_height, output_width,
                          min_queue_num, batch_size, reader_thread):
    """ load images and flow data simutiniously. """

    is_training = True if data_type is 'train' else False

    img1, img2, label, img1_path, img2_path = data_loader.load_pair_video_frame_from_text(
        data_path, channels, is_training, shuffle)

    img1 = tf.reshape(img1, shape=[raw_height, raw_width, channels * 3])
    img2 = tf.reshape(img2, shape=[raw_height, raw_width, channels * 3])

    img1 = preprocessing.factory.get_preprocessing(
        preprocessing_method1, data_type, img1,
        output_height, output_width, channels=channels * 3)

    img2 = preprocessing.factory.get_preprocessing(
        preprocessing_method2, data_type, img2,
        output_height, output_width, channels=channels * 3)

    return data_prefetch.generate_pair_imgs_labels_batch(
        img1, img2, label, img1_path, img2_path,
        shuffle, batch_size, min_queue_num, reader_thread)
