# -*- coding: utf-8 -*-
""" updated: 2017/3/16
"""
import tensorflow as tf

# classification
from gate.dataset.cifar import cifar10
from gate.dataset.cifar import cifar100
from gate.dataset.imagenet import imagenet
from gate.dataset.mnist import mnist
from gate.dataset.mnist import fashion_mnist
# depression
from gate.dataset.avec import avec2014
from gate.dataset.avec import avec2014_4view
from gate.dataset.avec import avec2013
from gate.dataset.avec import avec2013_4view
from gate.dataset.avec import cnu2017
# depression-aduio
from gate.dataset.avec import avec2014_audio
# depression-video
from gate.dataset.avec import avec2014_video
from gate.dataset.avec import avec2014_audio_rnn
# kinship
from gate.dataset.kinface import kinface2
from gate.dataset.kinface import kinface2_features
from gate.dataset.kinface import kinface2_5view_gc
# gan
from gate.dataset.celeba import celeba
# msceleb
from gate.dataset.msceleb import msceleb_align
from gate.dataset.lfw import lfw
# traffic flow
from gate.dataset.trafficflow import traffic_flow

dataset_map = {
    'mnist': mnist.mnist,
    'fashion_mnist': fashion_mnist.fashion_mnist,
    'cifar10': cifar10.cifar10,
    'cifar100': cifar100.cifar100,
    'imagenet': imagenet.imagenet,
    'avec2013': avec2013.avec2013,
    'avec2013_4view': avec2013_4view.avec2013_4view,
    'avec2014': avec2014.avec2014,
    'avec2014_video': avec2014_video.avec2014_video,
    'avec2014_audio': avec2014_audio.avec2014_audio,
    'avec2014_audio_rnn': avec2014_audio_rnn.avec2014_audio_rnn,
    'avec2014_4view': avec2014_4view.avec2014_4view,
    'cnu2017': cnu2017.cnu2017,
    'kinface2': kinface2.kinface2,
    'kinface2_feature': kinface2_features.kinface2_feature,
    'kinface2_5view_gc': kinface2_5view_gc.kinface2_5view_gc,
    'celeba':  celeba.celeba,
    'msceleb_align': msceleb_align.msceleb_align,
    'lfw': lfw.lfw,
    'traffic_flow': traffic_flow.traffic_flow
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