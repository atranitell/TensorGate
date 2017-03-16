# -*- coding: utf-8 -*-
""" updated: 2017/3/16
    Still have some problems
    Not to verify the result
"""

import os

import tensorflow as tf
from data import utils
from util_tools import output


class tfrecord():

    def __init__(self):
        pass

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def write_to_tfrecord(self, dataset):

        file_list_path = dataset.data_path
        total_num = dataset.total_num
        image_list, label_list, load_num = utils.read_from_file(file_list_path)

        if total_num != load_num:
            raise ValueError('Loading in %d images, but setting is %d images!'
                             %
                             (load_num, total_num))

        writer = tf.python_io.TFRecordWriter(dataset.name + '.tfrecords')

        director = output.progressive()

        with tf.Session() as sess:
            for idx in range(load_num):
                img = sess.run(tf.read_file(image_list[idx]))
                fname = bytes(image_list[idx], encoding='utf8')
                example = tf.train.Example(features=tf.train.Features(feature={
                    'label': self._int64_feature(label_list[idx]),
                    'image_raw': self._bytes_feature(img),
                    'filename': self._bytes_feature(fname)
                }))
                writer.write(example.SerializeToString())
                director.add_float(idx, load_num)

        writer.close()

    def read_from_tfrecord(self, data_path):

        file_suffix = os.path.splitext(data_path)[1]
        if file_suffix != '.tfrecords':
            raise ValueError('It is not a tfrecords file!')

        records_queue = tf.train.string_input_producer([data_path])
        reader = tf.TFRecordReader(data_path)
        _, serialized_example = reader.read(records_queue)

        features = tf.parse_single_example(
            serialized_example,
            features={
                'label': tf.FixedLenFeature([], tf.int64),
                'image_raw': tf.FixedLenFeature([], tf.string),
                'filename': tf.FixedLenFeature([], tf.string)
            })

        image = tf.image.decode_jpeg(features['image_raw'])
        label = tf.cast(features['label'], tf.int32)
        filename = tf.cast(features['filename'], tf.string)

        return image, label, filename
