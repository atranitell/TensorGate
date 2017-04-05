# -*- coding: utf-8 -*-
""" updated: 2017/04/05
"""

import os
from gate.utils import filesystem
from gate.utils import show


def parse_from_text(text_path, dtype_list, path_list):
    """ dtype_list is a tuple, which represent a list of data type.
        e.g. the file format like:
            a/1.jpg 3 2.5
            a/2.jpg 4 3.4
        dtype_list: (str, int, float)
        path_list: (true, false, false)

    Return:
        according to the dtype_list, return a tuple
        each item in tuple is a list.
    """
    # check path
    filesystem.raise_path_not_exists(text_path)
    dtype_size = len(dtype_list)
    assert dtype_size == len(path_list)

    # show
    show.SYS('Parse items from text file %s' % text_path)
    show.SYS('Items data type: ' + show.type_list_to_str(dtype_list))

    # construct the value to return and store
    res = []
    for _ in range(dtype_size):
        res.append([])

    # start to parse
    count = 0
    with open(text_path, 'r') as fp:
        for line in fp:
            # check content number
            r = line[:-1].split(' ')
            if len(r) != dtype_size:
                continue

            # check path
            # transfer type
            for idx, dtype in enumerate(dtype_list):
                val = dtype(r[idx])
                if path_list[idx]:
                    val = os.path.join(os.path.dirname(text_path), val)
                    filesystem.raise_path_not_exists(val)
                res[idx].append(val)

            # count
            count += 1

    show.INFO('Total loading in %d files.' % count)
    return res
