# -*- coding: utf-8 -*-
""" Contains the definition of a Dataset.

Functions:
    - images and labels preprocessing
    - from file/folder to load images and labels
    - generate parallel queue
    - batch for training or testing

Constant:
    - the total number of images
    - the input type 'train'/'test' in according to the different preprocessing method
    - datasets name
    - batch size

Call Method:
    main->datasets_factory->(cifar10/..)->dataset
"""
from abc import ABCMeta, abstractmethod

import tensorflow as tf
from gate import preprocessing
from gate.data import utils
from gate.data import dataset_tfrecords


class Dataset(metaclass=ABCMeta):
    """ Represents a Dataset specification.

    Note:
        any datasets('cifar','imagenet', and etc.) should inherit this class.
    """

    def __init__(self):
        pass

    @abstractmethod
    def loads(self):
        """ Should support at least a kind of reading method.
            Currently, we could load from *.txt file to acuqire image list and label,
            or we could load from *.tfrecords directly input all image and labels

            loads() function will includie preprocessing method.
            if you define the loads() exclusively, please put preprocessing in it.
        """
        pass

    def _load_data(self, data_load_method, data_path, total_num, shuffle):
        if data_load_method == 'text':
            return self._load_data_from_text(data_path, total_num, shuffle)

        elif data_load_method == 'tfrecord':
            return self._load_data_from_tfrecords(data_path)

        else:
            raise ValueError('Unknown data load method!')

    def _load_data_from_text(self, data_path, total_num, shuffle):
        file_list_path = data_path
        total_num = total_num
        image_list, label_list, load_num = utils.read_from_file(file_list_path)

        if total_num != load_num:
            raise ValueError('Loading in %d images, but setting is %d images!' %
                             (load_num, total_num))

        # construct a fifo queue
        images = tf.convert_to_tensor(image_list, dtype=tf.string)
        labels = tf.convert_to_tensor(label_list, dtype=tf.int32)
        input_queue = tf.train.slice_input_producer([images, labels], shuffle=shuffle)

        # preprocessing
        # there, the avec2014 image if 'JPEG' format
        image_raw = tf.read_file(input_queue[0])
        image_jpeg = tf.image.decode_jpeg(image_raw, channels=3)

        # image, label, filename
        return image_jpeg, input_queue[1], input_queue[0]

    def _load_data_from_tfrecords(self, data_path):
        dataset_tfrecord = dataset_tfrecords.tfrecord()
        # dataset_tfrecord.process(dataset
        return dataset_tfrecord.read_from_tfrecord(data_path)

    def _generate_image_label_batch(self, image, label, shuffle, min_queue_num,
                                    batch_size, reader_thread, filename=None):
        if shuffle:
            images, label_batch, filenames = tf.train.shuffle_batch(
                tensors=[image, label, filename],
                batch_size=batch_size,
                capacity=min_queue_num + 3 * batch_size,
                min_after_dequeue=min_queue_num,
                num_threads=reader_thread)
        else:
            images, label_batch, filenames = tf.train.batch(
                tensors=[image, label, filename],
                batch_size=batch_size,
                capacity=min_queue_num + 3 * batch_size,
                num_threads=reader_thread)

        return images, tf.reshape(label_batch, [batch_size]), filenames

    def _preprocessing_image(self, preprocessing_method, data_type,
                             image, output_height, output_width):
        return preprocessing.factory.get_preprocessing(
            preprocessing_method, data_type,
            image, output_height, output_width)

    def _preprocessing_label(self, label, data_type):
        if data_type is 'train':
            return self._preprocessing_label_for_train(label)
        elif data_type is 'test':
            return self._preprocessing_label_for_test(label)

    def _preprocessing_label_for_train(self, label):
        return label

    def _preprocessing_label_for_test(self, label):
        return label
