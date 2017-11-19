# -*- coding: utf-8 -*-
""" Save Image  - updated: 2017/05/09
"""

import tensorflow as tf
import scipy.misc
import numpy as np


def save_sampler(sess, sample_z, sample_y):
    """ Generate same sampler for various gan-family.
        in order to compare their performance.
    """
    with tf.Session() as sess:
        _z = sess.run(sample_z)
        _y = sess.run(sample_y)
        print(_z.shape)
        print(_z)
        print(_y.shape)
        for line in _y:
            print(line)
    # np.save(sample_z, sample_y)


def generate_all_class(num_class, y_dim, z_dim):
    """ generate one person with all label
    """
    # generate label
    y = [i for i in range(num_class)]
    sample_y = tf.convert_to_tensor(y, dtype=tf.int32)
    sample_y = tf.to_float(tf.one_hot(sample_y, depth=y_dim, on_value=1))

    # generate coder
    z = np.random.uniform(-1, 1, size=(1, z_dim))
    for i in range(num_class):
        z = np.stack((z, z), axis=0)
    print(z.shape)


def generate_sample(batchsize, n_kind, y_dim, z_dim):
    """ n_kind should less and equal to y_dim
    """
    if n_kind > y_dim:
        raise ValueError('n_kind larger than y_dim')

    z = np.random.uniform(-1, 1, size=(batchsize, z_dim))
    sample_z = tf.convert_to_tensor(z, dtype=tf.float32)

    n_row = batchsize // n_kind
    if n_row * n_kind is not batchsize:
        raise ValueError(
            'Error number of kind, row*kind not equal to batchsize')
    y = []
    for i in range(n_kind):
        for j in range(n_row):
            y.append(i)
    sample_y = tf.convert_to_tensor(y, dtype=tf.int32)
    sample_y = tf.to_float(tf.one_hot(sample_y, depth=y_dim, on_value=1))

    return sample_z, sample_y


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def inverse_transform(images):
    return (images + 1.) / 2.


def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
