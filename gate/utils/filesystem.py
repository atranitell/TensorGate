# -*- coding: utf-8 -*-
""" updated: 2017/3/28
"""

import os
from datetime import datetime


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