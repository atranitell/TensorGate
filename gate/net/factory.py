# -*- coding: utf-8 -*-
""" updated: 2017/3/16
"""

import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from gate.net import net_cifarnet
from gate.net import net_lenet
from gate.net import net_alexnet
from gate.net import net_resnet
from gate.net import net_resnet_v1
from gate.net import net_resnet_cifar
from gate.net import net_vgg
from gate.net import net_vgg_gap
from gate.net import net_inception_resnet_v2
from gate.net import net_inception_resnet_v1
from gate.net import net_lightnet
from gate.net import net_lightnet_bn
from gate.net import net_lightnet_56

networks_map = {
    'cifarnet': net_cifarnet.cifarnet(),
    'lenet': net_lenet.lenet(),
    'alexnet': net_alexnet.alexnet(),
    'alexnet_gap': net_alexnet.alexnet_gap(),
    'resnet_50': net_resnet.resnet_50(),
    'resnet_101': net_resnet.resnet_101(),
    'resnet_152': net_resnet.resnet_152(),
    'resnet_200': net_resnet.resnet_200(),
    'resnet_50_v1': net_resnet_v1.resnet_v1_50(),
    'resnet_101_v1': net_resnet_v1.resnet_v1_101(),
    'resnet_152_v1': net_resnet_v1.resnet_v1_152(),
    'resnet_200_v1': net_resnet_v1.resnet_v1_200(),
    'resnet_cifar': net_resnet_cifar.resnet_cifar(),
    'vgg_a': net_vgg.vgg_a(),
    'vgg_16': net_vgg.vgg_16(),
    'vgg_19': net_vgg.vgg_19(),
    'vgg_a_gap': net_vgg_gap.vgg_a(),
    'vgg_16_gap': net_vgg_gap.vgg_16(),
    'vgg_19_gap': net_vgg_gap.vgg_19(),
    'inception_resnet_v1': net_inception_resnet_v1.inception_resnet_v1(),
    'inception_resnet_v2': net_inception_resnet_v2.inception_resnet_v2(),
    'lightnet': net_lightnet.lightnet(),
    'lightnet_bn': net_lightnet_bn.lightnet_bn(),
    'lightnet_56': net_lightnet_56.lightnet_56()
}


def check_network(name, data_type):
    """ check network name """
    if name not in networks_map:
        raise ValueError('Unknown data_type %s' % data_type)
    if data_type == 'train':
        return True
    elif data_type == 'test':
        return False


def get_network(name, data_type, images, num_classes, name_scope='', reuse=False):
    """ get specified network """
    is_training = check_network(name, data_type)
    net = networks_map[name]
    with tf.variable_scope('net' + name_scope) as scope:
        if reuse:
            scope.reuse_variables()
        with arg_scope(net.arg_scope()):
            return net.model(images, num_classes, is_training)
