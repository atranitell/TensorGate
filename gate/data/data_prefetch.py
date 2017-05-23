# -*- coding: utf-8 -*-
""" updated: 2017/3/28
"""

import tensorflow as tf


def generate_batch(image, label, filename, shuffle,
                   batch_size, min_queue_num, reader_thread):
    """ for single image and label
    """
    if shuffle:
        images, labels, filenames = tf.train.shuffle_batch(
            tensors=[image, label, filename],
            batch_size=batch_size,
            capacity=min_queue_num + 3 * batch_size,
            min_after_dequeue=min_queue_num,
            num_threads=reader_thread)
    else:
        images, labels, filenames = tf.train.batch(
            tensors=[image, label, filename],
            batch_size=batch_size,
            capacity=min_queue_num + 3 * batch_size,
            num_threads=reader_thread)

    return images, tf.reshape(labels, [batch_size]), filenames


def generate_batch_multi_label(image, label, filename, shuffle, num_classes,
                               batch_size, min_queue_num, reader_thread):
    """ for single image and label
    """
    if shuffle:
        images, labels, filenames = tf.train.shuffle_batch(
            tensors=[image, label, filename],
            batch_size=batch_size,
            capacity=min_queue_num + 3 * batch_size,
            min_after_dequeue=min_queue_num,
            num_threads=reader_thread)
    else:
        images, labels, filenames = tf.train.batch(
            tensors=[image, label, filename],
            batch_size=batch_size,
            capacity=min_queue_num + 3 * batch_size,
            num_threads=reader_thread)

    return images, tf.reshape(labels, [batch_size, num_classes]), filenames


def generate_pair_batch(img1, img2, label, filename1, filename2, shuffle,
                        batch_size, min_queue_num, reader_thread):
    """ for pair images and same label
    """
    if shuffle:
        imgs1, imgs2, label_batch, filenames1, filenames2 = tf.train.shuffle_batch(
            tensors=[img1, img2, label, filename1, filename2],
            batch_size=batch_size,
            capacity=min_queue_num + 3 * batch_size,
            min_after_dequeue=min_queue_num,
            num_threads=reader_thread)
    else:
        imgs1, imgs2, label_batch, filenames1, filenames2 = tf.train.batch(
            tensors=[img1, img2, label, filename1, filename2],
            batch_size=batch_size,
            capacity=min_queue_num + 3 * batch_size,
            num_threads=reader_thread)

    return imgs1, imgs2, tf.reshape(label_batch, [batch_size]), filenames1, filenames2


def generate_pair_multiview_batch(
        X1, X2, Y1, Y2, label, filename1, filename2, shuffle,
        batch_size, min_queue_num, reader_thread):
    """ format, multi-input
        [x1_1, x1_2, ..., x1_n], [y1_1, y1_2, ..., y1_n], label, fname1, fname2
    """
    if shuffle:
        x1, x2, y1, y2, label_batch, filenames1, filenames2 = tf.train.shuffle_batch(
            tensors=[X1, X2, Y1, Y2, label, filename1, filename2],
            batch_size=batch_size,
            capacity=min_queue_num + 3 * batch_size,
            min_after_dequeue=min_queue_num,
            num_threads=reader_thread)
    else:
        x1, x2, y1, y2, label_batch, filenames1, filenames2 = tf.train.batch(
            tensors=[X1, X2, Y1, Y2, label, filename1, filename2],
            batch_size=batch_size,
            capacity=min_queue_num + 3 * batch_size,
            num_threads=reader_thread)

    return x1, x2, y1, y2, tf.reshape(label_batch, [batch_size]), filenames1, filenames2
