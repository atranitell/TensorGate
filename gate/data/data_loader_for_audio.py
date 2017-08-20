# -*- coding: utf-8 -*-
""" updated: 2017/06/13
"""

import random
import numpy as np
import tensorflow as tf

from gate.data import data_entry
from gate.data import data_prefetch
from gate.utils.logger import logger

STD = 0.001
NORM = 2


def _combine_block_continuous(filepath, start_idx, frames, length, invl):
    """
        channels: how many pictures will be compressed.
    """
    file_path_abs = str(filepath, encoding='utf-8')
    data = np.load(file_path_abs)

    valid_length = data.shape[0] - (invl * frames + length)
    if start_idx < 0:
        is_training = True
        start = random.randint(0, valid_length)
    else:
        is_training = False
        start = start_idx

    audio_data = np.array([])
    for i in range(frames):
        start_idx = start + i * invl
        _data = data[start_idx: start_idx + length]
        if is_training:
            _data += np.random.normal(STD, NORM)
        audio_data = np.append(audio_data, _data)

    audio_data = np.float32(np.reshape(audio_data, [frames, length]))
    return audio_data


def load_continuous_audio_from_npy(
        data_path, shuffle, data_type, audio,
        min_queue_num, batch_size, reader_thread):
    """ load audio sequence from a folder
    e.g. acquire audio file successently.
        a. |12345678901234567890|
        b.   |3456|   |2345|
    """
    res = data_entry.parse_from_text(
        data_path, (str, int, int), (True, False, False))
    files, starts, labels = res[0], res[1], res[2]

    logger.info('std:%f, norm%f' % (STD, NORM))

    # construct a fifo queue
    files = tf.convert_to_tensor(files, dtype=tf.string)
    starts = tf.convert_to_tensor(starts, dtype=tf.int32)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    filename, start, label = tf.train.slice_input_producer(
        [files, starts, labels], shuffle=shuffle)

    # combine
    content = tf.py_func(_combine_block_continuous,
                         [filename, start, audio.frames,
                          audio.frame_length, audio.frame_invl], tf.float32)

    content = tf.reshape(content, [audio.frames, audio.frame_length])

    return data_prefetch.generate_batch(
        content, label, filename, shuffle,
        batch_size, min_queue_num, reader_thread)
