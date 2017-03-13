# -*- coding: utf-8 -*-
""" ./datasets/datasets_factory.py
    A factory-pattern class which returns classification image/label pairs

    Tensorflow Version: r1.0
    Python Version: 3.5
    Update Date: 2017/03/13
    Author: Kai JIN
"""

from data import dataset_cifar10

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
    if name is 'cifar10':
        return dataset_cifar10.cifar10(data_type)
    else:
        raise ValueError('Name of dataset unknown %s' % name)