
import os
import numpy as np
import tensorflow as tf

from gate.utils.logger import logger
from gate.data.data_entry import parse_from_text
from gate.data import data_prefetch
from gate import preprocessing


def load_pair_numeric_data_from_npy(
        data_path, shuffle, data_type,
        min_queue_num, batch_size, reader_thread):
    """ load pair numeric data
    Format:
        data_path is a idx, it indicates the .npy path, have 3 line:
            pair1_path.npy
            pair2_path.npy
            label_path
        a dataset with .npy format, which shape is [N, D1, D2, ...]
        labels line with 'pair1_name pair2_name label'
        label with .npy format, which shape should be [N, L1]
    """
    with open(data_path) as fp:
        path_data_1 = fp.readline().split('\n')[0]
        path_data_2 = fp.readline().split('\n')[0]
        path_info = fp.readline().split('\n')[0]

    # from .npy load data
    basepath = os.path.dirname(data_path)
    data_1 = np.load(os.path.join(basepath, path_data_1))
    data_2 = np.load(os.path.join(basepath, path_data_2))

    # acquire filename and label info
    path_info = os.path.join(basepath, path_info)
    fnames_1, fnames_2, labels = parse_from_text(
        path_info, (str, str, int), (False, False, False))
    labels = np.reshape(np.array(labels), (data_1.shape[0], 1))

    # shape
    # logger.sys('data_1 shape is ' + str(data_1.shape))
    # logger.sys('data_2 shape is ' + str(data_2.shape))
    # logger.sys('label shape is ' + str(labels.shape))

    # construct a fifo queue
    data_1_list = tf.convert_to_tensor(data_1, dtype=tf.float32)
    data_2_list = tf.convert_to_tensor(data_2, dtype=tf.float32)
    label_list = tf.convert_to_tensor(labels, dtype=tf.int32)

    data_1, data_2, label, fname_1, fname_2 = tf.train.slice_input_producer(
        [data_1_list, data_2_list, label_list, fnames_1, fnames_2],
        shuffle=shuffle)

    return data_prefetch.generate_pair_batch(
        data_1, data_2, label, fname_1, fname_2, shuffle,
        batch_size, min_queue_num, reader_thread)


def load_pair_numeric_image_data(
        data_path, shuffle, data_type, image,
        min_queue_num, batch_size, reader_thread):
    """ load pair numeric and image data, e.g.:
        ImageX->Net->Feat->X1 |
                    DataX->X2 |
                              |->loss
        ImageY->Net->Feat->Y1 |
                    DataY->Y2 |

    Format:
        data_path is a idx, it indicates the .npy path, have 3 line:
            pair1_path.npy
            pair2_path.npy
            label_path
        a dataset with .npy format, which shape is [N, D1, D2, ...]
        labels line with 'pair1_name pair2_name label'
        label with .npy format, which shape should be [N, L1]
    """
    with open(data_path) as fp:
        path_data_1 = fp.readline().split('\n')[0]
        path_data_2 = fp.readline().split('\n')[0]
        path_info = fp.readline().split('\n')[0]

    # from .npy load data
    basepath = os.path.dirname(data_path)
    data_1 = np.load(os.path.join(basepath, path_data_1))
    data_2 = np.load(os.path.join(basepath, path_data_2))

    # acquire filename and label info
    path_info = os.path.join(basepath, path_info)
    fnames_1, fnames_2, labels = parse_from_text(
        path_info, (str, str, int), (True, True, False))
    labels = np.reshape(np.array(labels), (data_1.shape[0], 1))

    # shape
    # logger.sys('data_1 shape is ' + str(data_1.shape))
    # logger.sys('data_2 shape is ' + str(data_2.shape))
    # logger.sys('label shape is ' + str(labels.shape))

    # construct a fifo queue
    data_1_list = tf.convert_to_tensor(data_1, dtype=tf.float32)
    data_2_list = tf.convert_to_tensor(data_2, dtype=tf.float32)
    label_list = tf.convert_to_tensor(labels, dtype=tf.int32)

    data_1, data_2, label, fname_1, fname_2 = tf.train.slice_input_producer(
        [data_1_list, data_2_list, label_list, fnames_1, fnames_2],
        shuffle=shuffle)

    # preprocessing
    image_raw1 = tf.read_file(fname_1)
    image1 = tf.image.decode_image(image_raw1, channels=image.channels)
    image1 = tf.reshape(
        image1, [image.raw_height, image.raw_width, image.channels])

    image_raw2 = tf.read_file(fname_2)
    image2 = tf.image.decode_image(image_raw2, channels=image.channels)
    image2 = tf.reshape(
        image2, [image.raw_height, image.raw_width, image.channels])

    # image, label, filename
    image1 = preprocessing.factory.get_preprocessing(
        image.preprocessing_method1, data_type, image1,
        image.output_height, image.output_width)

    image2 = preprocessing.factory.get_preprocessing(
        image.preprocessing_method2, data_type, image2,
        image.output_height, image.output_width)

    return data_prefetch.generate_pair_multiview_batch(
        image1, data_1, image2, data_2, label, fname_1, fname_2,
        shuffle, batch_size, min_queue_num, reader_thread)
