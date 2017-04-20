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
    imgpath, label = tf.train.slice_input_producer([image_list, label_list], shuffle=shuffle)

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
