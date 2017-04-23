# -*- coding: utf-8 -*-
""" updated: 2017/3/28
"""

import numpy as np
import tensorflow as tf

from PIL import Image

from gate import preprocessing
from gate.data import data_entry
from gate.data import data_prefetch
from gate.utils import filesystem


def load_image_from_text(
        data_path, shuffle, data_type,
        frames, channels, preprocessing_method,
        raw_height, raw_width, output_height, output_width,
        min_queue_num, batch_size, reader_thread):
    """ a normal loader method from text to parse content
    Format:
        path label
        path-to-fold/img0 0
        path-to-fold/img1 10
    """
    res = data_entry.parse_from_text(data_path, (str, int), (True, False))
    image_list = res[0]
    label_list = res[1]

    # construct a fifo queue
    image_list = tf.convert_to_tensor(image_list, dtype=tf.string)
    label_list = tf.convert_to_tensor(label_list, dtype=tf.int32)
    imgpath, label = tf.train.slice_input_producer(
        [image_list, label_list], shuffle=shuffle)

    # preprocessing
    image_raw = tf.read_file(imgpath)
    image = tf.image.decode_image(image_raw, channels=3)
    image = tf.reshape(image, [raw_height, raw_width, channels])

    # image, label, filename
    image = preprocessing.factory.get_preprocessing(
        preprocessing_method, data_type, image,
        output_height, output_width)

    return data_prefetch.generate_batch(
        image, label, imgpath, shuffle,
        batch_size, min_queue_num, reader_thread)


def load_pair_image_from_text(
        data_path, shuffle, data_type, frames, channels, 
        preprocessing_method1, preprocessing_method2,
        raw_height, raw_width, output_height, output_width,
        min_queue_num, batch_size, reader_thread):
    """ load pair data
    Format:
        img_0_1 img0_2 0
        img_1_1 img1_2 10
    """
    res = data_entry.parse_from_text(
        data_path, (str, str, int), (True, True, False))
    image1_list = res[0]
    image2_list = res[1]
    label_list = res[2]

    # construct a fifo queue
    image1_list = tf.convert_to_tensor(image1_list, dtype=tf.string)
    image2_list = tf.convert_to_tensor(image2_list, dtype=tf.string)
    label_list = tf.convert_to_tensor(label_list, dtype=tf.int32)
    imgpath1, imgpath2, label = tf.train.slice_input_producer(
        [image1_list, image2_list, label_list], shuffle=shuffle)

    # preprocessing
    image_raw1 = tf.read_file(imgpath1)
    image1 = tf.image.decode_image(image_raw1, channels=3)
    image1 = tf.reshape(image1, [raw_height, raw_width, channels])

    image_raw2 = tf.read_file(imgpath2)
    image2 = tf.image.decode_image(image_raw2, channels=3)
    image2 = tf.reshape(image2, [raw_height, raw_width, channels])

    # image, label, filename
    image1 = preprocessing.factory.get_preprocessing(
        preprocessing_method1, data_type, image1,
        output_height, output_width)

    image2 = preprocessing.factory.get_preprocessing(
        preprocessing_method2, data_type, image2,
        output_height, output_width)

    return data_prefetch.generate_pair_batch(
        image1, image2, label, imgpath1, imgpath2, shuffle,
        batch_size, min_queue_num, reader_thread)


def load_image_from_memory(
        data_path, shuffle, data_type,
        frames, channels, preprocessing_method,
        raw_height, raw_width, output_height, output_width,
        min_queue_num, batch_size, reader_thread):
    """ The function will construct a database to store in memory,
            in order to reduce the time of visiting the disk.
    Format:
        path idx label
        path-to-fold/img0 0
        path-to-fold/img1 10
    """
    res = data_entry.parse_from_text(data_path, (str, int), (True, False))
    img_list, label_list = res[0], res[1]

    database = []
    for i, img_path in enumerate(img_list):
        filesystem.raise_path_not_exists(img_path)
        img_content = np.asarray(Image.open(img_path))
        img_content = np.reshape(
            img_content, (img_content.shape[0], img_content.shape[1], channels))
        database.append(img_content)

    # convert to tensor
    img_list = tf.convert_to_tensor(img_list, dtype=tf.string)
    label_list = tf.convert_to_tensor(label_list, dtype=tf.int32)
    database = np.array(database, dtype=np.uint8)

    # construct a fifo queue
    path, label, image = tf.train.slice_input_producer(
        [img_list, label_list, database], shuffle=shuffle)

    # preprocessing
    image = preprocessing.factory.get_preprocessing(
        preprocessing_method, data_type, image,
        output_height, output_width)

    return data_prefetch.generate_batch(
        image, label, path, shuffle,
        batch_size, min_queue_num, reader_thread)
