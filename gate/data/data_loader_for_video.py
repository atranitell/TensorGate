# -*- coding: utf-8 -*-
""" updated: 2017/3/28
"""

import os
import numpy as np
import math
import random
import tensorflow as tf

from PIL import Image

from gate import preprocessing
from gate.data import data_entry
from gate.data import data_prefetch
from gate.utils import filesystem


def combine_images_block_random(foldpath, frames, is_training):
    """
        frames: how many pictures will be compressed.
    """
    fold_path_abs = str(foldpath, encoding='utf-8')
    img_list = [fs for fs in os.listdir(fold_path_abs) if len(fs.split('.jpg')) > 1]

    # generate idx without reptitious
    # img_indice = random.sample([i for i in range(len(img_list))], frames)
    invl = math.floor(len(img_list) / float(frames))
    start = 0
    img_indice = []
    for _ in range(frames):
        end = start + invl
        if is_training:
            img_indice.append(random.randint(start, end - 1))
        else:
            img_indice.append(start)
        start = end

    # generate
    img_selected_list = []
    for idx in range(frames):
        img_path = os.path.join(fold_path_abs, img_list[img_indice[idx]])
        img_selected_list.append(img_path)
    img_selected_list.sort()

    # compression to (256,256,3*16)
    combine = np.asarray(Image.open(img_selected_list[0]))
    for idx, img in enumerate(img_selected_list):
        if idx == 0:
            continue
        img_content = np.asarray(Image.open(img))
        combine = np.dstack((combine, img_content))

    return combine


def load_block_random_video_from_text(
        data_path, shuffle, data_type,
        frames, channels, preprocessing_method,
        raw_height, raw_width, output_height, output_width,
        min_queue_num, batch_size, reader_thread):
    """ load video sequence from a folder
    e.g.
        a. |12345|12345|12345|12345|12345|12345|...
        b. | 2   |1    |   4 |  3  |1    |  3  |...
        c. 214313 = 6*3channels = 18 channels

    Format:
        path-of-fold1 0
        path-of-fold2 10

    args:
        channels: how much images in a folder will be compress into a image.
    """
    # parse from folder list
    res = data_entry.parse_from_text(data_path, (str, int), (True, False))
    folds, labels = res[0], res[1]

    # construct a fifo queue
    folds = tf.convert_to_tensor(folds, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    foldname, label = tf.train.slice_input_producer([folds, labels], shuffle=shuffle)

    # combine to one
    is_training = True if data_type == 'train' else False
    image = tf.py_func(combine_images_block_random,
                       [foldname, frames, is_training], tf.uint8)
    image = tf.reshape(image, shape=[raw_height, raw_width, frames * channels])

    # preprocess
    image = preprocessing.factory.get_preprocessing(
        preprocessing_method, data_type, image,
        output_height, output_width, channels=frames * channels)

    return data_prefetch.generate_batch(
        image, label, foldname, shuffle,
        batch_size, min_queue_num, reader_thread)


def combine_images_block_continuous(foldpath, start_idx, frames):
    """
        channels: how many pictures will be compressed.
    """
    fold_path_abs = str(foldpath, encoding='utf-8')
    img_list = [fs for fs in os.listdir(fold_path_abs) if len(fs.split('.jpg')) > 1]

    # for train
    if start_idx < 0:
        start = random.randint(0, len(img_list) - frames)
    # for test
    else:
        start = start_idx

    # generate
    img_selected_list = []
    for idx in range(frames):
        img_path = os.path.join(fold_path_abs, img_list[start + idx])
        img_selected_list.append(img_path)
    img_selected_list.sort()

    # compression to (256,256,3*16)
    combine = np.asarray(Image.open(img_selected_list[0]))
    for idx, img in enumerate(img_selected_list):
        if idx == 0:
            continue
        img_content = np.asarray(Image.open(img))
        combine = np.dstack((combine, img_content))

    return combine


def load_block_continuous_video_from_text(
        data_path, shuffle, data_type,
        frames, channels, preprocessing_method,
        raw_height, raw_width, output_height, output_width,
        min_queue_num, batch_size, reader_thread):
    """ load video sequence from a folder
    e.g. acquire video frame successently.
        a. |12345678901234567890|
        b.   |3456|   |2345|

    Format:
        path start_point label
        path-of-fold1 0 0
        path-of-fold2 1 10
    """
    res = data_entry.parse_from_text(data_path, (str, int, int), (True, False, False))
    folds, starts, labels = res[0], res[1], res[2]

    # construct a fifo queue
    folds = tf.convert_to_tensor(folds, dtype=tf.string)
    starts = tf.convert_to_tensor(starts, dtype=tf.int32)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    foldname, start, label = tf.train.slice_input_producer([folds, starts, labels], shuffle=shuffle)

    # combine
    image = tf.py_func(combine_images_block_continuous,
                       [foldname, start, frames], tf.uint8)
    image = tf.reshape(image, shape=[raw_height, raw_width, channels * 3])

    image = preprocessing.factory.get_preprocessing(
        preprocessing_method, data_type, image,
        output_height, output_width, channels=channels * 3)

    return data_prefetch.generate_batch(
        image, label, foldname, shuffle,
        batch_size, min_queue_num, reader_thread)


def combine_pair_images_block_random(img_fold, flow_fold, frames=16, is_training=True):
    """
        assemble images into pair sequence data
    """
    img_fold_path_abs = str(img_fold, encoding='utf-8')
    flow_fold_path_abs = str(flow_fold, encoding='utf-8')

    img_list = [fs for fs in os.listdir(img_fold_path_abs) if len(fs.split('.jpg')) > 1]
    flow_list = [fs for fs in os.listdir(flow_fold_path_abs) if len(fs.split('.jpg')) > 1]

    # pay attention, please keep the image and flow images
    #   is same number(frame id) in same people in a folder
    invl = math.floor(len(img_list) / float(frames))
    start = 0
    indice = []

    # for trainset, random to choose None
    # for testset, choose fixed point
    for _ in range(frames):
        end = start + invl
        if is_training:
            indice.append(random.randint(start, end - 1))
        else:
            indice.append(start)
        start = end

    # acquire actual image path according to indice
    img_selected_list = []
    flow_selected_list = []
    for idx in range(frames):
        img_path = os.path.join(img_fold_path_abs, img_list[indice[idx]])
        flow_path = os.path.join(flow_fold_path_abs, flow_list[indice[idx]])

        img_selected_list.append(img_path)
        flow_selected_list.append(flow_path)

    img_selected_list.sort()
    flow_selected_list.sort()

    # combine frames into one image
    combine_img = np.asarray(Image.open(img_selected_list[0]))
    combine_flow = np.asarray(Image.open(flow_selected_list[0]))
    for idx in range(frames):
        if idx == 0:
            continue
        img_content = np.asarray(Image.open(img_selected_list[idx]))
        flow_content = np.asarray(Image.open(flow_selected_list[idx]))
        combine_img = np.dstack((combine_img, img_content))
        combine_flow = np.dstack((combine_flow, flow_content))

    return combine_img, combine_flow


def load_pair_block_random_video_from_text(
        data_path, shuffle, data_type, frames, channels,
        preprocessing_method1, preprocessing_method2,
        raw_height, raw_width, output_height, output_width,
        min_queue_num, batch_size, reader_thread):
    """ load images and flow data simutiniously.
        Reading a pair of folds and labels from text file.
        The function will server for same label but has different style.
        Attention: the function will load in fold path not images

    Format:
        path-of-fold1-1 path-of-fold1-2 0
        path-of-fold2-1 path-of-fold2-2 10
    """

    is_training = True if data_type is 'train' else False

    res = data_entry.parse_from_text(data_path, (str, str, int), (True, True, False))
    fold_1, fold_2, labels = res[0], res[1], res[2]

    fold_1 = tf.convert_to_tensor(fold_1, dtype=tf.string)
    fold_2 = tf.convert_to_tensor(fold_2, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    fold_1_path, fold_2_path, label = tf.train.slice_input_producer(
        [fold_1, fold_2, labels], shuffle=shuffle)

    img1, img2 = tf.py_func(combine_pair_images_block_random,
                            [fold_1_path, fold_2_path, frames, is_training],
                            [tf.uint8, tf.uint8])

    img1 = tf.reshape(img1, shape=[raw_height, raw_width, channels * frames])
    img2 = tf.reshape(img2, shape=[raw_height, raw_width, channels * frames])

    img1 = preprocessing.factory.get_preprocessing(
        preprocessing_method1, data_type, img1,
        output_height, output_width, channels=channels * frames)

    img2 = preprocessing.factory.get_preprocessing(
        preprocessing_method2, data_type, img2,
        output_height, output_width, channels=channels * frames)

    return data_prefetch.generate_pair_batch(
        img1, img2, label, fold_1_path, fold_2_path,
        shuffle, batch_size, min_queue_num, reader_thread)
