# -*- coding: utf-8 -*-
""" updated: 2017/3/28
"""

import os
import numpy as np
import tensorflow as tf

from PIL import Image

from gate import preprocessing
from gate.data import data_entry
from gate.data import data_prefetch
from gate.utils import filesystem


def load_image_from_text(
        data_path, shuffle, data_type, image,
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
    image_content = tf.image.decode_image(image_raw, channels=image.channels)
    if image.raw_height > 0:
        image_content = tf.reshape(
            image_content, [image.raw_height, image.raw_width, image.channels])

    # image, label, filename
    image_content = preprocessing.factory.get_preprocessing(
        image.preprocessing_method1, data_type, image_content,
        image.output_height, image.output_width)

    return data_prefetch.generate_batch(
        image_content, label, imgpath, shuffle,
        batch_size, min_queue_num, reader_thread)


def load_image_5view_gc_from_text(
        data_path, shuffle, data_type, image,
        min_queue_num, batch_size, reader_thread):
    """ for kinship 5view + geometry contrain
    """
    res = data_entry.parse_from_text(
        data_path, (str, str, int), (False, False, False))
    image_list1 = res[0]
    image_list2 = res[1]
    label_list = res[2]

    def path_to_tensor(image_list, prefix):
        data = []
        for i in range(len(image_list)):
            path = prefix + image_list[i]
            filesystem.raise_path_not_exists(path)
            data.append(path)
        result = tf.convert_to_tensor(data, dtype=tf.string)
        return result

    _dir = os.path.dirname(data_path)

    img_list_full1 = path_to_tensor(image_list1, _dir + '\\full\\')
    img_list_full2 = path_to_tensor(image_list2, _dir + '\\full\\')
    img_list_leye1 = path_to_tensor(image_list1, _dir + '\\left_eye\\')
    img_list_leye2 = path_to_tensor(image_list2, _dir + '\\left_eye\\')
    img_list_reye1 = path_to_tensor(image_list1, _dir + '\\right_eye\\')
    img_list_reye2 = path_to_tensor(image_list2, _dir + '\\right_eye\\')
    img_list_nose1 = path_to_tensor(image_list1, _dir + '\\nose\\')
    img_list_nose2 = path_to_tensor(image_list2, _dir + '\\nose\\')
    img_list_mouth1 = path_to_tensor(image_list1, _dir + '\\mouth\\')
    img_list_mouth2 = path_to_tensor(image_list2, _dir + '\\mouth\\')

    # ------------------------------------

    res = data_entry.parse_from_text(
        _dir + '\\distances.txt',
        (str, float, float, float, float, float, float, float, float),
        (False, False, False, False, False, False, False, False, False))

    img_gc1 = []
    img_gc2 = []
    for i in range(len(image_list1)):
        for j, value in enumerate(res[0]):
            if os.path.basename(image_list1[i]) == os.path.basename(value):
                dist = []
                for k in range(1, 9):
                    dist.append(float(res[k][j]))
                img_gc1.append(dist)
            if os.path.basename(image_list2[i]) == os.path.basename(value):
                dist = []
                for k in range(1, 9):
                    dist.append(float(res[k][j]))
                img_gc2.append(dist)

    img_list_gc1 = tf.convert_to_tensor(img_gc1, dtype=tf.float32)
    img_list_gc2 = tf.convert_to_tensor(img_gc2, dtype=tf.float32)
    label_list = tf.convert_to_tensor(label_list, dtype=tf.int32)

    # ------------------------------------

    res_slice = tf.train.slice_input_producer(
        [img_list_full1, img_list_full2, img_list_leye1, img_list_leye2,
         img_list_reye1, img_list_reye2, img_list_nose1, img_list_nose2,
         img_list_mouth1, img_list_mouth2,
         img_list_gc1, img_list_gc2, label_list], shuffle=shuffle)

    imgpath_f1, imgpath_f2 = res_slice[0], res_slice[1]
    imgpath_le1, imgpath_le2 = res_slice[2], res_slice[3]
    imgpath_re1, imgpath_re2 = res_slice[4], res_slice[5]
    imgpath_n1, imgpath_n2 = res_slice[6], res_slice[7]
    imgpath_m1, imgpath_m2 = res_slice[8], res_slice[9]
    img_gc1, img_gc2, label = res_slice[10], res_slice[11], res_slice[12]

    # preprocessing
    def preprocessing_img(imgpath):
        _image = tf.read_file(imgpath)
        _image = tf.image.decode_image(_image, channels=image.channels)
        _image = tf.reshape(
            _image, [image.raw_height, image.raw_width, image.channels])
        return preprocessing.factory.get_preprocessing(
            image.preprocessing_method1, data_type, _image,
            image.output_height, image.output_width)

    img_full1 = preprocessing_img(imgpath_f1)
    img_full2 = preprocessing_img(imgpath_f2)
    img_leye1 = preprocessing_img(imgpath_le1)
    img_leye2 = preprocessing_img(imgpath_le2)
    img_reye1 = preprocessing_img(imgpath_re1)
    img_reye2 = preprocessing_img(imgpath_re2)
    img_nose1 = preprocessing_img(imgpath_n1)
    img_nose2 = preprocessing_img(imgpath_n2)
    img_mouth1 = preprocessing_img(imgpath_m1)
    img_mouth2 = preprocessing_img(imgpath_m2)

    return data_prefetch.generate_5view_gc_batch(
        img_full1, img_full2, img_leye1, img_leye2,
        img_reye1, img_reye2, img_nose1, img_nose2,
        img_mouth1, img_mouth2, img_gc1, img_gc2,
        label, imgpath_f1, imgpath_f2, shuffle,
        batch_size, min_queue_num, reader_thread)


def load_image_4view_from_text(
        data_path, shuffle, data_type, image,
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

    image_list1 = res[0].copy()
    image_list2 = res[0].copy()
    image_list3 = res[0].copy()
    for _i in range(len(image_list)):
        image_list1[_i] = image_list1[_i].replace('frames', 'top_frames')
        image_list2[_i] = image_list2[_i].replace('frames', 'middle_frames')
        image_list3[_i] = image_list3[_i].replace('frames', 'bottom_frames')

    # construct a fifo queue
    image_list = tf.convert_to_tensor(image_list, dtype=tf.string)
    image1_list = tf.convert_to_tensor(image_list1, dtype=tf.string)
    image2_list = tf.convert_to_tensor(image_list2, dtype=tf.string)
    image3_list = tf.convert_to_tensor(image_list3, dtype=tf.string)
    label_list = tf.convert_to_tensor(label_list, dtype=tf.int32)
    imgpath, imgpath1, imgpath2, imgpath3, label = tf.train.slice_input_producer(
        [image_list, image_list1, image_list2, image_list3, label_list], shuffle=shuffle)

    # preprocessing
    image_raw = tf.read_file(imgpath)
    image_content = tf.image.decode_image(image_raw, channels=image.channels)
    image_content = tf.reshape(
        image_content, [image.raw_height, image.raw_width, image.channels])
    image_content = preprocessing.factory.get_preprocessing(
        image.preprocessing_method1, data_type, image_content,
        image.output_height, image.output_width)

    # preprocessing
    image_raw1 = tf.read_file(imgpath1)
    image_content1 = tf.image.decode_image(image_raw1, channels=image.channels)
    image_content1 = tf.reshape(image_content1, [156, 156, 3])
    image_content1 = preprocessing.factory.get_preprocessing(
        image.preprocessing_method2, data_type, image_content1,
        image.output_height, image.output_width)

    # preprocessing
    image_raw2 = tf.read_file(imgpath2)
    image_content2 = tf.image.decode_image(image_raw2, channels=image.channels)
    image_content2 = tf.reshape(image_content2, [156, 156, 3])
    image_content2 = preprocessing.factory.get_preprocessing(
        image.preprocessing_method2, data_type, image_content2,
        image.output_height, image.output_width)

    # preprocessing
    image_raw3 = tf.read_file(imgpath3)
    image_content3 = tf.image.decode_image(image_raw3, channels=image.channels)
    image_content3 = tf.reshape(image_content3, [156, 156, 3])
    image_content3 = preprocessing.factory.get_preprocessing(
        image.preprocessing_method2, data_type, image_content3,
        image.output_height, image.output_width)

    return data_prefetch.generate_4view_batch(
        image_content, image_content1, image_content2, image_content3,
        label, imgpath, shuffle,
        batch_size, min_queue_num, reader_thread)


def load_image_from_text_multi_label(
        data_path, shuffle, data_type, num_classes, image,
        min_queue_num, batch_size, reader_thread):
    """ load image with multi-label data
    Format:
        path label1 label2, label3, ..., labeln
        path-to-fold/img0 0 1 0 0 0 1...
        path-to-fold/img1 0 0 1 0 0 1...
    """
    type_label = (str, )
    type_path = (True, )
    for idx_type in range(num_classes):
        type_label += (int, )
        type_path += (False, )
    res = data_entry.parse_from_text(data_path, type_label, type_path)
    image_list = res[0]

    # exchange channels
    label_list = []
    for idx_pp in range(len(res[0])):
        label = []
        for idx_label in range(num_classes):
            label.append(res[idx_label + 1][idx_pp])
        label_list.append(label)

    # construct a fifo queue
    image_list = tf.convert_to_tensor(image_list, dtype=tf.string)
    label_list = tf.convert_to_tensor(label_list, dtype=tf.int32)
    imgpath, label = tf.train.slice_input_producer(
        [image_list, label_list], shuffle=shuffle)

    # preprocessing
    image_raw = tf.read_file(imgpath)
    image_content = tf.image.decode_image(image_raw, channels=image.channels)
    image_content = tf.reshape(
        image_content, [image.raw_height, image.raw_width, image.channels])

    # image, label, filename
    image_content = preprocessing.factory.get_preprocessing(
        image.preprocessing_method1, data_type, image_content,
        image.output_height, image.output_width)

    return data_prefetch.generate_batch_multi_label(
        image_content, label, imgpath, shuffle, num_classes,
        batch_size, min_queue_num, reader_thread)


def load_pair_image_from_text(
        data_path, shuffle, data_type, image,
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
    image1 = tf.image.decode_image(image_raw1, channels=image.channels)
    image1 = tf.reshape(
        image1, [image.raw_height, image.raw_width, image.channels])

    image_raw2 = tf.read_file(imgpath2)
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

    return data_prefetch.generate_pair_batch(
        image1, image2, label, imgpath1, imgpath2, shuffle,
        batch_size, min_queue_num, reader_thread)


def _handcrafted_feature_extract(image, feature_type=None):

    if feature_type is None:
        return image
    else:
        _type = str(feature_type, encoding="utf-8")
        if _type in ['LBP', 'lbp']:
            img = image
            # img = handcrafted.LBP(image)
            img = np.float32(np.reshape(img, (img.shape[0], img.shape[1], 1)))
            return img
        else:
            raise ValueError('Error input with feature type.')


def load_pair_image_from_text_with_multiview(
        data_path, shuffle, data_type, image,
        min_queue_num, batch_size, reader_thread):
    """ load pair data
    Format:
        img_x1 img_y1 0
        img_x2 img_y2 10

    e.g.
        for x -> preprocessing -> distorted image (x1_1)
                                       |-> handcrafted image (x1_2)
        for y -> preprocessing -> distorted image (y1_1)
                                       |-> handcrafted image (y1_2)
        net input:
        x1_1 -> net1 |
                     | -> _x1 |
        x1_2 -> net2 |        |
                              | -> metric distance
        y1_1 -> net1'|        |
                     | -> _y1 |
        y1_2 -> net2'|

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
    image1 = tf.image.decode_image(image_raw1, channels=image.channels)
    image1 = tf.reshape(
        image1, [image.raw_height, image.raw_width, image.channels])

    image_raw2 = tf.read_file(imgpath2)
    image2 = tf.image.decode_image(image_raw2, channels=image.channels)
    image2 = tf.reshape(
        image2, [image.raw_height, image.raw_width, image.channels])

    # image, label, filename
    imageX_1 = preprocessing.factory.get_preprocessing(
        image.preprocessing_method1, data_type, image1,
        image.output_height, image.output_width)

    imageY_1 = preprocessing.factory.get_preprocessing(
        image.preprocessing_method2, data_type, image2,
        image.output_height, image.output_width)

    # extract handcrafted feature
    imageX_2 = tf.py_func(_handcrafted_feature_extract,
                          [imageX_1, 'LBP'], [tf.float32])
    imageX_2 = tf.to_float(tf.reshape(
        imageX_2, [image.raw_height * image.raw_width * 1]))

    imageY_2 = tf.py_func(_handcrafted_feature_extract,
                          [imageY_1, 'LBP'], [tf.float32])
    imageY_2 = tf.to_float(tf.reshape(
        imageY_2, [image.raw_height * image.raw_width * 1]))

    return data_prefetch.generate_pair_multiview_batch(
        imageX_1, imageX_2, imageY_1, imageY_2,
        label, imgpath1, imgpath2, shuffle,
        batch_size, min_queue_num, reader_thread)


def load_image_from_memory(
        data_path, shuffle, data_type, image,
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
            img_content, (img_content.shape[0],
                          img_content.shape[1],
                          image.channels))
        database.append(img_content)

    # convert to tensor
    img_list = tf.convert_to_tensor(img_list, dtype=tf.string)
    label_list = tf.convert_to_tensor(label_list, dtype=tf.int32)
    database = np.array(database, dtype=np.uint8)

    # construct a fifo queue
    path, label, image_content = tf.train.slice_input_producer(
        [img_list, label_list, database], shuffle=shuffle)

    # preprocessing
    image_content = preprocessing.factory.get_preprocessing(
        image.preprocessing_method1, data_type, image_content,
        image.output_height, image.output_width)

    return data_prefetch.generate_batch(
        image_content, label, path, shuffle,
        batch_size, min_queue_num, reader_thread)


def _load_image_from_npy(filepath, channels):
    """ load image from npy file.
    """
    file_path_abs = str(filepath, encoding='utf-8')
    data = np.load(file_path_abs)

    data = np.float32(np.reshape(data, [112, 112, channels]))
    return data


def load_image_from_npy(data_path, shuffle, data_type, image,
                        min_queue_num, batch_size, reader_thread):
    """ load image
    """
    res = data_entry.parse_from_text(data_path, (str, int), (True, False))
    files, labels = res[0], res[1]

    # construct a fifo queue
    files = tf.convert_to_tensor(files, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    filename, label = tf.train.slice_input_producer(
        [files, labels], shuffle=shuffle)

    # combine
    content = tf.py_func(_load_image_from_npy,
                         [filename, image.channels], tf.float32)

    content = tf.reshape(
        content, [image.output_height, image.output_width, image.channels])

    print(content)

    return data_prefetch.generate_batch(
        content, label, filename, shuffle,
        batch_size, min_queue_num, reader_thread)
