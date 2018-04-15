# -*- coding: utf-8 -*-
""" Contains a factory for building various models.
    Author: Kai JIN
    Updated: 2017-08-28
"""
import tensorflow as tf
from core.network import model

arg_scope = tf.contrib.framework.arg_scope

network_map = {
    'alexnet': model._alexnet,
    'cifarnet': model._cifarnet,
    'overfeat': model._overfeat,
    'vgg_a': model._vgg,
    'vgg_16': model._vgg,
    'vgg_19': model._vgg,
    'inception_v1': model._inception_v1,
    'inception_v2': model._inception_v2,
    'inception_v3': model._inception_v3,
    'inception_v4': model._inception_v4,
    'inception_resnet_v1': model._inception_resnet_v1,
    'inception_resnet_v2': model._inception_resnet_v2,
    'lenet': model._lenet,
    'resnet_v1_50': model._resnet,
    'resnet_v1_101': model._resnet,
    'resnet_v1_152': model._resnet,
    'resnet_v1_200': model._resnet,
    'resnet_v2_50': model._resnet,
    'resnet_v2_101': model._resnet,
    'resnet_v2_152': model._resnet,
    'resnet_v2_200': model._resnet,
    'mobilenet_v1': model._mobilenet_v1,
    'nasnet_cifar': model._nasnet,
    'nasnet_mobile': model._nasnet,
    'nasnet_large': model._nasnet,
    'squeezenet': model._squeezenet,
    'simplenet': model._simplenet,
    'mlp': model._mlp,
    'audionet': model._audionet,
    'resnet_v2_50_bishared': model._bisahred_resnet,
    'resnet_v2_101_bishared': model._bisahred_resnet,
    'resnet_v2_152_bishared': model._bisahred_resnet,
    'resnet_v2_200_bishared': model._bisahred_resnet
}

argscope_map = {
    'alexnet': model._alexnet_scope,
    'cifarnet': model._cifarnet_scope,
    'overfeat': model._overfeat_scope,
    'vgg_a': model._vgg_scope,
    'vgg_16': model._vgg_scope,
    'vgg_19': model._vgg_scope,
    'inception_v1': model._inception_scope,
    'inception_v2': model._inception_scope,
    'inception_v3': model._inception_scope,
    'inception_v4': model._inception_scope,
    'inception_resnet_v1': None,
    'inception_resnet_v2': model._inception_resnet_v2_scope,
    'lenet': model._lenet_scope,
    'resnet_v1_50': model._resnet_scope,
    'resnet_v1_101': model._resnet_scope,
    'resnet_v1_152': model._resnet_scope,
    'resnet_v1_200': model._resnet_scope,
    'resnet_v2_50': model._resnet_scope,
    'resnet_v2_101': model._resnet_scope,
    'resnet_v2_152': model._resnet_scope,
    'resnet_v2_200': model._resnet_scope,
    'mobilenet_v1': model._mobilenet_v1_scope,
    'nasnet_cifar': model._nasnet_scope,
    'nasnet_mobile': model._nasnet_scope,
    'nasnet_large': model._nasnet_scope,
    'squeezenet': None,
    'simplenet': None,
    'mlp': None,
    'audionet': None,
    'resnet_v2_50_bishared': model._bisahred_resnet_scope,
    'resnet_v2_101_bishared': model._bisahred_resnet_scope,
    'resnet_v2_152_bishared': model._bisahred_resnet_scope,
    'resnet_v2_200_bishared': model._bisahred_resnet_scope
}


def network(X, config, phase, name='', reuse=None):
  """ net factory
  """
  is_training = True if phase == 'train' else False
  with tf.variable_scope(name) as scope:
    argscope = argscope_map[config.net.name]
    if reuse:
      scope.reuse_variables()
    if argscope is not None:
      with arg_scope(argscope(config)):
        return network_map[config.net.name](X, config, is_training)
    else:
      return network_map[config.net.name](X, config, is_training)
