# -*- coding: utf-8 -*-
""" updated: 2017/3/16
"""
import tensorflow as tf

# classification
from gate.dataset.cifar import cifar10
from gate.dataset.cifar import cifar100
from gate.dataset.imagenet import imagenet
from gate.dataset.mnist import mnist
# depression
from gate.dataset.avec import avec2014
from gate.dataset.avec import avec2014_4view
# kinship
from gate.dataset.kinface import kinface2
from gate.dataset.kinface import kinface2_features


dataset_map = {
    'mnist': mnist.mnist,
    'cifar10': cifar10.cifar10,
    'cifar100': cifar100.cifar100,
    'imagenet': imagenet.imagenet,
    'avec2014': avec2014.avec2014,
    'avec2014_4view': avec2014_4view.avec2014_4view,
    'kinface2': kinface2.kinface2,
    'kinface2_feature': kinface2_features.kinface2_feature
}


def get_dataset(name, data_type, chkp_path=None):
    """Given a dataset name and a data_type returns a Dataset.
    Args:
        name: String, the name of the dataset.
        data_type: A split name. e.g. 'train', 'val', 'test'
    Returns:
        A `Dataset` class.
    Raises:
        ValueError: If the dataset `name` is unknown.
    """
    if name not in dataset_map:
        raise ValueError('Unknown dataset %s' % name)
    with tf.name_scope(name):
        return dataset_map[name](data_type, name, chkp_path)
