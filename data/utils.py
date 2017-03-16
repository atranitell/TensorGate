# -*- coding: utf-8 -*-
""" updated: 2017/3/16
"""

import os
from datetime import datetime


def read_from_file(file_list):
    """ Reading images and labels from text file.
        The format like:
            path-to-fold/img0 0
            path-to-fold/img1 10
            ...
    """
    if not os.path.isfile(file_list):
        raise ValueError('Wrong file path %s', file_list)
    imglist = []
    labels = []
    root = os.path.dirname(file_list)
    count = 0
    with open(file_list, 'r') as fp:
        for line in fp:
            subpath, label = line[:-1].split(' ')
            # some image path could not find
            actual_path = os.path.join(root, subpath)
            if not os.path.isfile(actual_path):
                continue
                # raise ValueError('File: %s could not find.' % imglist[-1])
            imglist.append(actual_path)
            labels.append(int(label))
            count = count + 1

    return imglist, labels, count


def dir_log_constructor(folder_path):
    """ Add a date suffix to avoid to conflict with previous training model

    Args:
        folder_path: the prefix of folder_path

    Returns:
        dir_log: the actual folder path
    """
    date_str = datetime.strftime(datetime.now(), '%Y%m%d%H%M')
    dir_log = folder_path + '_' + date_str
    if not os.path.exists(dir_log):
        os.mkdir(dir_log)
    return dir_log
