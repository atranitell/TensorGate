# -*- coding: utf-8 -*-
""" updated: 2017/3/28
    a data loader factory, to avoid much more library for dataset
"""
from gate.data import data_loader_for_image
from gate.data import data_loader_for_video
from gate.data import data_loader_for_numeric
from gate.data import data_loader_for_audio


def load_image_from_text(
        data_path, shuffle, data_type, image,
        min_queue_num, batch_size, reader_thread):
    return data_loader_for_image.load_image_from_text(
        data_path, shuffle, data_type, image,
        min_queue_num, batch_size, reader_thread)


def load_image_4view_from_text(
        data_path, shuffle, data_type, image,
        min_queue_num, batch_size, reader_thread):
    return data_loader_for_image.load_image_4view_from_text(
        data_path, shuffle, data_type, image,
        min_queue_num, batch_size, reader_thread)


def load_image_from_text_multi_label(
        data_path, shuffle, data_type, num_classes, image,
        min_queue_num, batch_size, reader_thread):
    return data_loader_for_image.load_image_from_text_multi_label(
        data_path, shuffle, data_type, num_classes, image,
        min_queue_num, batch_size, reader_thread)


def load_pair_image_from_text(
        data_path, shuffle, data_type, image,
        min_queue_num, batch_size, reader_thread):
    return data_loader_for_image.load_pair_image_from_text(
        data_path, shuffle, data_type, image,
        min_queue_num, batch_size, reader_thread)


def load_pair_image_from_text_with_multiview(
        data_path, shuffle, data_type, image,
        min_queue_num, batch_size, reader_thread):
    return data_loader_for_image.load_pair_image_from_text_with_multiview(
        data_path, shuffle, data_type, image,
        min_queue_num, batch_size, reader_thread)


def load_image_from_memory(
        data_path, shuffle, data_type, image,
        min_queue_num, batch_size, reader_thread):
    return data_loader_for_image.load_image_from_memory(
        data_path, shuffle, data_type, image,
        min_queue_num, batch_size, reader_thread)


def load_block_random_video_from_text(
        data_path, shuffle, data_type, image,
        min_queue_num, batch_size, reader_thread):
    return data_loader_for_video.load_block_random_video_from_text(
        data_path, shuffle, data_type, image,
        min_queue_num, batch_size, reader_thread)


def load_block_continuous_video_from_text(
        data_path, shuffle, data_type, image,
        min_queue_num, batch_size, reader_thread):
    return data_loader_for_video.load_block_continuous_video_from_text(
        data_path, shuffle, data_type, image,
        min_queue_num, batch_size, reader_thread)


def load_pair_block_continuous_video_from_text(
        data_path, shuffle, data_type, image,
        min_queue_num, batch_size, reader_thread):
    return data_loader_for_video.load_block_continuous_video_from_text(
        data_path, shuffle, data_type, image,
        min_queue_num, batch_size, reader_thread)


def load_pair_numeric_data_from_npy(
        data_path, shuffle, data_type,
        min_queue_num, batch_size, reader_thread):
    return data_loader_for_numeric.load_pair_numeric_data_from_npy(
        data_path, shuffle, data_type,
        min_queue_num, batch_size, reader_thread)


def load_pair_numeric_image_data(
        data_path, shuffle, data_type, image,
        min_queue_num, batch_size, reader_thread):
    return data_loader_for_numeric.load_pair_numeric_image_data(
        data_path, shuffle, data_type, image,
        min_queue_num, batch_size, reader_thread)


def load_continuous_audio_from_npy(
        data_path, shuffle, data_type, audio,
        min_queue_num, batch_size, reader_thread):
    return data_loader_for_audio.load_continuous_audio_from_npy(
        data_path, shuffle, data_type, audio,
        min_queue_num, batch_size, reader_thread)
