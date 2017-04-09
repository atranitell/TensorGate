# -*- coding: utf-8 -*-
""" updated: 2017/3/28
"""

import os
import shutil
from datetime import datetime


def copy_folder(src_path, dst_path):
    """ copy a folder, including all files and sub-folder to dst
        if dst_path does not exist, it will create a folder
    """
    raise_path_not_exists(src_path)
    shutil.copytree(src_path, dst_path)


def create_folder_with_date(folder_path):
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


def raise_path_not_exists(path):
    if not os.path.exists(path):
        raise ValueError('Path could not find in %s' % path)
