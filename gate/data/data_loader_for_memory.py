# -*- coding: utf-8 -*-
""" updated: 2017/3/30

    The module mainly served for the data loaded in memory
    in order to boost the I/O performance compared with
    load from text, which load file according to path in text.

    The function will load all images in memory, and stored as
    dict(database) to be called by upper class.

    performance test result:

"""

import os
import math
import random

import numpy as np
from PIL import Image

from gate.utils import filesystem
from gate.utils import Progressive

from gate.data import data_entry


def add_all_image_from_fold_to_database(database, fold_path, label):
    """ !in place operation
    """
    fold_name = os.path.basename(fold_path)
    if database.has_key(fold_name):
        print('[WARN] key %s has in database.' % fold_name)
        return

    database[fold_name] = {}
    for img_name in sorted(os.listdir(fold_path)):
        img_path = os.path.join(fold_path, img_name)
        filesystem.raise_path_not_exists(img_path)

        img_content = np.asarray(Image.open(img_path))

        database[fold_name][img_name] = {}
        database[fold_name][img_name]['content'] = img_content
        database[fold_name][img_name]['label'] = label
        database[fold_name][img_name]['path'] = img_path


def create_single_video_database_in_memory(data_path, database=None):
    """ the data_path should be a text file, e.g.
        folder1 label1
        folder2 label2
        folder3 label3

    Return: a dict like:
        database[fold_idx][img_idx][multi_pose_idx]
        database[0][0][0][‘content’], the numpy image
        database[0][0][0][‘label’], the class/regression value
        database[0][0][0][‘path’], the actual path in disk          
    """
    # load folder information
    folds, labels = data_entry.read_fold_from_text_list_with_label(data_path)
    print('[INFO] System has loading database %s', data_path)

    # traverse every folder
    database = {}
    progress_bar = Progressive(min_scale=2.0)
    count = 0

    # start to process
    for idx_fold, fold_path in enumerate(folds):
        add_all_image_from_fold_to_database(database, fold_path, labels[idx_fold])
        count += 1
        progress_bar.add_float(idx_fold, len(folds))

    print('[INFO] Total load %d folds to memory.' % count)

    return database


def create_pair_video_database_in_memory(data_path):
    """ the data_path should be a text file, e.g.
        /usr/data/feat1/class1 /usr/data/feat2/class1 label1
        /usr/data/feat1/class2 /usr/data/feat2/class2 label2

        pay attention:
            we expect same label have same class folder name.
            'feat1/class1' and 'feat2/class1' have same 'class1'
            sub folder name.

    Return: a dict like:
        database[fold_idx][img_idx][multi_pose_idx]
    """
    # load folder information
    folds_0, folds_1, labels = data_entry.read_pair_folds_from_text_list_with_label(data_path)
    print('[INFO] System has loading database %s', data_path)
    
    # traverse every folder
    database = {}
    progress_bar = Progressive(min_scale=2.0)
    count = 0

    # start to process
    for idx, _ in enumerate(folds_0):

        add_all_image_from_fold_to_database(database, folds_0[idx], labels[idx])
        add_all_image_from_fold_to_database(database, folds_1[idx], labels[idx])
        
        count += 2
        progress_bar.add_float(idx, len(folds_0))
    
    print('[INFO] Total load %d folds to memory.' % count)

    return database