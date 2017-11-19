# -*- coding: utf-8 -*-
""" parse log file
    Updated: 2017/06/11
"""
import os
import re


def bigram(filepath, phase, key):
    """
    All input will be not case-sensitive
        phase and key should be same line.
        each line should has key iter

    Args:
        filepath: the path to log file.
        phase: [TRN], [TST], [VAL].
        key: like loss, mae, rmse, error
    """
    if not os.path.exists(filepath):
        raise ValueError('File could not find in %s' % filepath)

    # transfer to lower case
    phase = phase.lower()
    key = key.lower()

    # return data
    data = {}
    data['iter'] = []
    data[key] = []

    # parse
    with open(filepath, 'r') as fp:
        for line in fp:
            line = line.lower()
            if line.find(phase) < 0:
                continue
            r_iter = re.findall('iter:(.*?),', line)
            r_key = re.findall(key + ':(.*?),', line)
            if len(r_key) == 0:
                r_key = re.findall(key + ':(.*).', line)
            if len(r_iter) and len(r_key):
                data['iter'].append(int(r_iter[0]))
                data[key].append(float(r_key[0]))

    # check equal
    assert len(data['iter']) == len(data[key])
    return data
