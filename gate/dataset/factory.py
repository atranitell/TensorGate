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
from gate.dataset.avec import avec2013
from gate.dataset.avec import cnu2017
# depression-aduio
from gate.dataset.avec import avec2014_audio
# depression-video
from gate.dataset.avec import avec2014_video
# kinship
from gate.dataset.kinface import kinface2
from gate.dataset.kinface import kinface2_features
# gan
from gate.dataset.celeba import celeba
# msceleb
from gate.dataset.msceleb import msceleb_align

dataset_map = {
    'mnist': mnist.mnist,
    'cifar10': cifar10.cifar10,
    'cifar100': cifar100.cifar100,
    'imagenet': imagenet.imagenet,
    'avec2014': avec2014.avec2014,
    'avec2013': avec2013.avec2013,
    'avec2014_video': avec2014_video.avec2014_video,
    'avec2014_audio': avec2014_audio.avec2014_audio,
    'cnu2017': cnu2017.cnu2017,
    'avec2014_4view': avec2014_4view.avec2014_4view,
    'kinface2': kinface2.kinface2,
    'kinface2_feature': kinface2_features.kinface2_feature,
    'celeba':  celeba.celeba,
    'msceleb_align': msceleb_align.msceleb_align
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
