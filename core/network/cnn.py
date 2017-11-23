# -*- coding: utf-8 -*-
""" Contains a factory for building various models.
    Author: Kai JIN
    Updated: 2017-08-28
"""

import sys
sys.path.append('core/network/')

import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from core.network.nets import alexnet
from core.network.nets import cifarnet
from core.network.nets import inception
from core.network.nets import lenet
from core.network.nets import mobilenet_v1
from core.network.nets import overfeat
from core.network.nets import resnet_v1
from core.network.nets import resnet_v2
from core.network.nets import vgg
from core.network.nets.nasnet import nasnet


def slim_cifarnet(X, config, is_training):
  """ the config is config[task][net]
  """
  argscope = cifarnet.cifarnet_arg_scope(config.net.weight_decay)
  net = cifarnet.cifarnet(X, num_classes=config.data.num_classes,
                          is_training=is_training,
                          dropout_keep_prob=config.net.dropout_keep)
  return net, argscope


networks_map = {
    'cifarnet': slim_cifarnet
}


def network(X, config, scope='', reuse=None):
  """ net factory
  """
  is_training = True if config.phase == 'train' else False
  # get network and argscope
  net, argscope = networks_map[config.net.name](X, config, is_training)
  with tf.variable_scope(scope) as scope:
    if reuse:
      scope.reuse_variables()
    with arg_scope(argscope):
      return net
