# -*- coding: utf-8 -*-
""" ./datasets/datasets_factory.py
    A factory-pattern class which returns classification image/label pairs

    Tensorflow Version: r1.0
    Python Version: 3.5
    Update Date: 2017/03/13
    Author: Kai JIN
"""

from data import dataset_cifar10
from data import dataset_avec2014
from data import dataset_avec2014_flow

dataset_map = {
    'cifar10': dataset_cifar10.cifar10,
    'avec2014': dataset_avec2014.avec2014,
    'avec2014_flow': dataset_avec2014_flow.avec2014_flow
}

def check_dataset(name):
    if name not in dataset_map:
        raise ValueError('Unknown dataset %s' % name)

def get_dataset(name, data_type):
    """Given a dataset name and a data_type returns a Dataset.

    Args:
        name: String, the name of the dataset.
        data_type: A split name. e.g. 'train', 'test'

    Returns:
        A `Dataset` class.

    Raises:
        ValueError: If the dataset `name` is unknown.
    """
    check_dataset(name)
    return dataset_map[name](data_type)