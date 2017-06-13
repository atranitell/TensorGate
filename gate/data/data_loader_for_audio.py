# -*- coding: utf-8 -*-
""" updated: 2017/06/13
"""

import os
import numpy as np
import tensorflow as tf

from gate.data import data_entry
from gate.data import data_prefetch


def _combine_block_continuous(foldpath, start_idx, frames):
    """
        channels: how many pictures will be compressed.
    """
    fold_path_abs = str(foldpath, encoding='utf-8')
    _list = [fs for fs in os.listdir(
        fold_path_abs) if len(fs.split('.npy')) > 1]

    # generate
    _selected_list = []
    for idx in range(frames):
        _path = os.path.join(fold_path_abs, _list[start_idx + idx])
        _selected_list.append(_path)
    _selected_list.sort()

    # compression to (frames, length)
    combine = np.load(_selected_list[0])
    for idx, file in enumerate(_selected_list):
        if idx == 0:
            continue
        combine = np.column_stack((combine, np.load(file)))
    combine = np.float32(np.transpose(combine, (1, 0)))

    return combine


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
    folds, starts, labels = res[0], res[1], res[2]

    # construct a fifo queue
    folds = tf.convert_to_tensor(folds, dtype=tf.string)
    starts = tf.convert_to_tensor(starts, dtype=tf.int32)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    foldname, start, label = tf.train.slice_input_producer(
        [folds, starts, labels], shuffle=shuffle)

    # combine
    content = tf.py_func(_combine_block_audio_continuous,
                         [foldname, start, audio.frames], tf.float32)

    content = tf.reshape(content, [audio.frames, audio.frame_length])

    return data_prefetch.generate_batch(
        content, label, foldname, shuffle,
        batch_size, min_queue_num, reader_thread)
