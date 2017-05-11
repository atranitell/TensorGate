# -*- coding: utf-8 -*-
""" updated: 2017/3/28
    a data loader factory, to avoid much more library for dataset
"""
from gate.data import data_loader_for_image
from gate.data import data_loader_for_video


def load_image_from_text(
        data_path, shuffle, data_type,
        frames, channels, preprocessing_method,
        raw_height, raw_width, output_height, output_width,
        min_queue_num, batch_size, reader_thread):
    return data_loader_for_image.load_image_from_text(
        data_path, shuffle, data_type,
        frames, channels, preprocessing_method,
        raw_height, raw_width, output_height, output_width,
        min_queue_num, batch_size, reader_thread)


def load_image_from_text_multi_label(
        data_path, shuffle, data_type, num_classes,
        frames, channels, preprocessing_method,
        raw_height, raw_width, output_height, output_width,
        min_queue_num, batch_size, reader_thread):
    return data_loader_for_image.load_image_from_text_multi_label(
        data_path, shuffle, data_type, num_classes,
        frames, channels, preprocessing_method,
        raw_height, raw_width, output_height, output_width,
        min_queue_num, batch_size, reader_thread)


def load_pair_image_from_text(
        data_path, shuffle, data_type, frames, channels,
        preprocessing_method1, preprocessing_method2,
        raw_height, raw_width, output_height, output_width,
        min_queue_num, batch_size, reader_thread):
    return data_loader_for_image.load_pair_image_from_text(
        data_path, shuffle, data_type, frames, channels,
        preprocessing_method1, preprocessing_method2,
        raw_height, raw_width, output_height, output_width,
        min_queue_num, batch_size, reader_thread)


def load_image_from_memory(
        data_path, shuffle, data_type,
        frames, channels, preprocessing_method,
        raw_height, raw_width, output_height, output_width,
        min_queue_num, batch_size, reader_thread):
    return data_loader_for_image.load_image_from_memory(
        data_path, shuffle, data_type,
        frames, channels, preprocessing_method,
        raw_height, raw_width, output_height, output_width,
        min_queue_num, batch_size, reader_thread)


def load_block_random_video_from_text(
        data_path, shuffle, data_type,
        frames, channels, preprocessing_method,
        raw_height, raw_width, output_height, output_width,
        min_queue_num, batch_size, reader_thread):
    return data_loader_for_video.load_block_random_video_from_text(
        data_path, shuffle, data_type,
        frames, channels, preprocessing_method,
        raw_height, raw_width, output_height, output_width,
        min_queue_num, batch_size, reader_thread)


def load_block_continuous_video_from_text(
        data_path, shuffle, data_type,
        frames, channels, preprocessing_method,
        raw_height, raw_width, output_height, output_width,
        min_queue_num, batch_size, reader_thread):
    return data_loader_for_video.load_block_continuous_video_from_text(
        data_path, shuffle, data_type,
        frames, channels, preprocessing_method,
        raw_height, raw_width, output_height, output_width,
        min_queue_num, batch_size, reader_thread)


def load_pair_block_succ_video_from_text(
        data_path, shuffle, data_type, frames, channels,
        preprocessing_method1, preprocessing_method2,
        raw_height, raw_width, output_height, output_width,
        min_queue_num, batch_size, reader_thread):
    return data_loader_for_video.load_pair_block_succ_video_from_text(
        data_path, shuffle, data_type, frames, channels,
        preprocessing_method1, preprocessing_method2,
        raw_height, raw_width, output_height, output_width,
        min_queue_num, batch_size, reader_thread)
