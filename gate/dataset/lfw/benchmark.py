# -*- coding: utf-8 -*-
""" updated: 2017/07/06
"""

import numpy as np
import sys

def benchmark(data_path):
    # load data
    features = np.load(data_path)

    # divide to 10 fold
    fold = {}
    for idx in range(10):
        pass


def test(name, chkp_path, layer_name):
    print(name, chkp_path, layer_name)


def pipline(name, chkp_path, fn):
    print(fn)
    fn(name, chkp_path, fn)

if __name__ == '__main__':
    # data_path = sys.argv[1]
    # benchmark(data_path)
    pipline('1', '2', test(123, 456, 789))
