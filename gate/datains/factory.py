# -*- coding: utf-8 -*-
""" updated: 2017/3/16
"""

from gate.datains import dataset_cifar10
from gate.datains import dataset_avec2014
from gate.datains import dataset_avec2014_16f
from gate.datains import dataset_avec2014_flow
from gate.datains import dataset_avec2014_flow_16f
from gate.datains import dataset_avec2014_fuse_16f
from gate.datains import dataset_mnist

dataset_map = {
    'mnist': dataset_mnist.mnist,
    'cifar10': dataset_cifar10.cifar10,
    'avec2014': dataset_avec2014.avec2014,
    'avec2014_16f': dataset_avec2014_16f.avec2014_16f,
    'avec2014_flow': dataset_avec2014_flow.avec2014_flow,
    'avec2014_flow_16f': dataset_avec2014_flow_16f.avec2014_flow_16f,
    'avec2014_fuse_16f': dataset_avec2014_fuse_16f.avec2014_fuse_16f
}


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

    if name not in dataset_map:
        raise ValueError('Unknown dataset %s' % name)
    return dataset_map[name](data_type, name)
