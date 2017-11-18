# -*- coding: utf-8 -*-
""" offer a series of data processing method.
    like smooth, downsample.
    Updated: 2017/06/11
"""
import os
import math
import json


def downsampling(data, interval):
    """ data is a list.
        interval is a int type number.
        return a new list with downsampling
    """
    length = len(data)
    assert interval > 0
    ret = []
    for idx in range(0, length, interval):
        ret.append(data[idx])
    return ret


def downsampling_bigram(data, interval):
    """ a simple interface for data like:
        data['iter'] = [...]
        data[key] = [...]
        it will travser all key in data and to downsampling
    """
    ret = {}
    for var in data:
        ret[var] = downsampling(data[var], interval)
    return ret


def smooth(data, num):
    """ K means
    """
    ret = []
    mid = math.floor(num / 2.0)
    for i in range(len(data)):
        if i > mid and i < len(data) - num:
            avg = 0
            for j in range(num):
                avg = avg + data[i + j]
            ret.append(avg / num)
        else:
            ret.append(data[i])
    return ret


def write_to_text(data, keys, filepath):
    """ data should be a dict
        keys is a key in dict
    ps:
        all key should has number of element in data.
        len(data[key0]) == len(data[key1]) == ...
    """
    for key in keys:
        if key not in data:
            raise ValueError('%s not in data' % key)

    fw = open(filepath, 'w')
    for idx in range(len(data[keys[0]])):
        line = ''
        for key in keys:
            line += str(data[key][idx]) + ' '
        line += '\n'
        fw.write(line)
    fw.close()


def json_parser(filepath):
    """ parse json file to dict
    """
    if not os.path.isfile(filepath):
        raise ValueError('File could not find in %s' % filepath)
    with open(filepath, 'r') as fp:
        config = json.load(fp)
    return config
