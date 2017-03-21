# -*- coding: utf-8 -*-
""" updated: 2017/3/16
"""

from tensorflow.contrib.framework import arg_scope
from nets import net_cifarnet
from nets import net_lenet
from nets import net_alexnet
from nets import net_resnet
from nets import net_vgg
from nets import net_inception_resnet_v2
from nets import net_lightnet
from nets import net_lightnet_bn

networks_map = {
    'cifarnet': net_cifarnet.cifarnet(),
    'lenet': net_lenet.lenet(),
    'alexnet': net_alexnet.alexnet(),
    'resnet_50': net_resnet.resnet_50(),
    'resnet_101': net_resnet.resnet_101(),
    'resnet_152': net_resnet.resnet_152(),
    'resnet_200': net_resnet.resnet_200(),
    'vgg_a': net_vgg.vgg_a(),
    'vgg_16': net_vgg.vgg_16(),
    'vgg_19': net_vgg.vgg_19(),
    'inception_resnet_v2': net_inception_resnet_v2.inception_resnet_v2(),
    'lightnet': net_lightnet.lightnet(),
    'lightnet_bn': net_lightnet_bn.lightnet_bn()
}


def check_network(name, data_type):
    """ check network name """
    if name not in networks_map:
        raise ValueError('Unknown data_type %s' % data_type)
    if data_type == 'train':
        return True
    elif data_type == 'test':
        return False


def get_network(name, data_type, images, num_classes):
    """ get specified network """
    is_training = check_network(name, data_type)
    net = networks_map[name]
    with arg_scope(net.arg_scope()):
        return net.model(images, num_classes, is_training)
