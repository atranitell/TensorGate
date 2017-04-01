# -*- coding: utf-8 -*-
""" updated: 2017/3/28
"""

import os
from gate.utils import filesystem


def read_image_from_text_list_with_label(filepath):
    """ Reading images and labels from text file.

        The format like:
            path-to-fold/img0 0
            path-to-fold/img1 10
            ...
    """
    # check path
    filesystem.raise_path_not_exists(filepath)

    imgs = []
    labels = []
    count = 0

    with open(filepath, 'r') as fp:
        for line in fp:
            subpath, label = line[:-1].split(' ')
            # some image path could not find
            actual_path = os.path.join(os.path.dirname(filepath), subpath)
            if not os.path.isfile(actual_path):
                continue
            imgs.append(actual_path)
            labels.append(int(label))
            count += 1

    print('[INFO] Total load %d files.' % count)

    return imgs, labels, count


def read_fold_from_text_list_with_label(filepath):
    """ Reading folds and labels from text file.
        Attention: the function will load in fold path not images

        The format like:
            path-of-fold1 0
            path-of-fold2 10
    """
    # check path
    filesystem.raise_path_not_exists(filepath)

    fold_0 = []
    labels = []

    count = 0
    with open(filepath, 'r') as fp:
        for line in fp:

            r = line.split(' ')
            if len(r) <= 1:
                continue

            r[0] = os.path.join(os.path.dirname(filepath), r[0])
            filesystem.raise_path_not_exists(r[0])

            fold_0.append(r[0])
            labels.append(int(r[1]))

            count += 1

    print('[INFO] Total load %d folds.' % count)

    return fold_0, labels


def read_fold_from_text_list_with_label_succ(filepath):
    """ Reading folds and labels from text file.
        Attention: the function will load in fold path not images

        The format like:
            path start_point label
            path-of-fold1 0 0
            path-of-fold2 1 10
    """
    # check path
    filesystem.raise_path_not_exists(filepath)

    fold_0 = []
    starts = []
    labels = []

    count = 0
    with open(filepath, 'r') as fp:
        for line in fp:

            r = line.split(' ')
            if len(r) <= 1:
                continue

            r[0] = os.path.join(os.path.dirname(filepath), r[0])
            filesystem.raise_path_not_exists(r[0])

            fold_0.append(r[0])
            starts.append(int(r[1]))
            labels.append(int(r[2]))

            count += 1

    print('[INFO] Total load %d folds.' % count)

    return fold_0, starts, labels


def read_pair_folds_from_text_list_with_label(filepath):
    """ Reading a pair of folds and labels from text file.
        The function will server for same label but has different style.
        Attention: the function will load in fold path not images

        The format like:
            path-of-fold1-1 path-of-fold1-2 0
            path-of-fold2-1 path-of-fold2-2 10
    """
    # check path
    filesystem.raise_path_not_exists(filepath)

    fold_0 = []
    fold_1 = []
    labels = []

    count = 0
    with open(filepath, 'r') as fp:
        for line in fp:

            r = line.split(' ')
            if len(r) <= 1:
                continue

            r[0] = os.path.join(os.path.dirname(filepath), r[0])
            r[1] = os.path.join(os.path.dirname(filepath), r[1])

            filesystem.raise_path_not_exists(r[0])
            filesystem.raise_path_not_exists(r[1])

            fold_0.append(r[0])
            fold_1.append(r[1])
            labels.append(int(r[2]))

            count += 1

    print('[INFO] Total load %d pair folds.' % count)

    return fold_0, fold_1, labels
