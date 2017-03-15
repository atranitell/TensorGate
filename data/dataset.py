# -*- coding: utf-8 -*-
# ./data/datasets_data_provider.py
#
#    Tensorflow Version: r1.0
#    Python Version: 3.5
#    Update Date: 2017/03/13
#    Author: Kai JIN
# ==============================================================================

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
from preprocessing import preprocessing_factory
import tensorflow as tf


class Dataset(metaclass=ABCMeta):
    """ Represents a Dataset specification.

    Note:
        any datasets('cifar','imagenet', and etc.) should inherit this class.
    """

    def __init__(self):
        pass

    @abstractmethod
    def loads(self):
        pass

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
        return preprocessing_factory.get_preprocessing(
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
