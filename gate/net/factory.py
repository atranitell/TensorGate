# -*- coding: utf-8 -*-
"""
DECTRION FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2018/4/19

--------------------------------------------------------

NETWORK FACTORY: Parse parameters from the config

"""

import sys
sys.path.append('gate\\net')

import tensorflow as tf
from gate.net import models

arg_scope = tf.contrib.framework.arg_scope

network_map = {
    # 'alexnet': model._alexnet,
    'cifarnet': models._cifarnet,
    # 'overfeat': model._overfeat,
    # 'vgg_a': model._vgg,
    # 'vgg_16': model._vgg,
    # 'vgg_19': model._vgg,
    # 'inception_v1': model._inception_v1,
    # 'inception_v2': model._inception_v2,
    # 'inception_v3': model._inception_v3,
    # 'inception_v4': model._inception_v4,
    # 'inception_resnet_v1': model._inception_resnet_v1,
    # 'inception_resnet_v2': model._inception_resnet_v2,
    'lenet': models._lenet,
    'resnet_v1_50': models._resnet,
    'resnet_v1_101': models._resnet,
    'resnet_v1_152': models._resnet,
    'resnet_v1_200': models._resnet,
    'resnet_v2_50': models._resnet,
    'resnet_v2_101': models._resnet,
    'resnet_v2_152': models._resnet,
    'resnet_v2_200': models._resnet,
    # 'mobilenet_v1': model._mobilenet_v1,
    # 'nasnet_cifar': model._nasnet,
    # 'nasnet_mobile': model._nasnet,
    # 'nasnet_large': model._nasnet,
    # 'squeezenet': model._squeezenet,
    # 'simplenet': model._simplenet,
    # 'mlp': model._mlp,
    # 'audionet': model._audionet,
    # 'resnet_v2_50_bishared': model._bisahred_resnet,
    # 'resnet_v2_101_bishared': model._bisahred_resnet,
    # 'resnet_v2_152_bishared': model._bisahred_resnet,
    # 'resnet_v2_200_bishared': model._bisahred_resnet
}

argscope_map = {
    # 'alexnet': model._alexnet_scope,
    'cifarnet': models._cifarnet_scope,
    # 'overfeat': model._overfeat_scope,
    # 'vgg_a': model._vgg_scope,
    # 'vgg_16': model._vgg_scope,
    # 'vgg_19': model._vgg_scope,
    # 'inception_v1': model._inception_scope,
    # 'inception_v2': model._inception_scope,
    # 'inception_v3': model._inception_scope,
    # 'inception_v4': model._inception_scope,
    # 'inception_resnet_v1': None,
    # 'inception_resnet_v2': model._inception_resnet_v2_scope,
    'lenet': models._lenet_scope,
    'resnet_v1_50': models._resnet_scope,
    'resnet_v1_101': models._resnet_scope,
    'resnet_v1_152': models._resnet_scope,
    'resnet_v1_200': models._resnet_scope,
    'resnet_v2_50': models._resnet_scope,
    'resnet_v2_101': models._resnet_scope,
    'resnet_v2_152': models._resnet_scope,
    'resnet_v2_200': models._resnet_scope,
    # 'mobilenet_v1': model._mobilenet_v1_scope,
    # 'nasnet_cifar': model._nasnet_scope,
    # 'nasnet_mobile': model._nasnet_scope,
    # 'nasnet_large': model._nasnet_scope,
    # 'squeezenet': None,
    # 'simplenet': None,
    # 'mlp': None,
    # 'audionet': None,
    # 'resnet_v2_50_bishared': model._bisahred_resnet_scope,
    # 'resnet_v2_101_bishared': model._bisahred_resnet_scope,
    # 'resnet_v2_152_bishared': model._bisahred_resnet_scope,
    # 'resnet_v2_200_bishared': model._bisahred_resnet_scope
}


def get_net(X, config, phase, name='', reuse=None):
  """
  """
  is_training = True if phase == 'train' else False
  with tf.variable_scope(name) as scope:
    argscope = argscope_map[config.name]
    if reuse:
      scope.reuse_variables()
    if argscope is not None:
      with arg_scope(argscope(config)):
        return network_map[config.name](X, config, is_training)
    else:
      return network_map[config.name](X, config, is_training)


# def resnet_v2_50_argscope(config):
#   return resnet_v2.resnet_arg_scope(
#       weight_decay=config.net.weight_decay,
#       batch_norm_decay=config.net.batch_norm_decay,
#       batch_norm_epsilon=config.net.batch_norm_epsilon,
#       batch_norm_scale=config.net.batch_norm_scale,
#       activation_fn=config.net.activation_fn,
#       use_batch_norm=config.net.use_batch_norm)


# def resnet_v2_50(config, inputs, is_training, scope='', reuse=None):
#   with tf.variable_scope(scope) as scope:
#     argscope = resnet_v2_50_argscope(config)
#     with tf.contrib.framework.arg_scope(argscope):
#       logit, end_points = resnet_v2.resnet_v2_50(
#           inputs=inputs,
#           num_classes=config.net.num_classes,
#           is_training=is_training,
#           global_pool=config.net.global_pool,
#           output_stride=config.net.output_stride,
#           spatial_squeeze=config.net.spatial_squeeze,
#           reuse=reuse,
#           scope=config.net.scope)
#       return logit, end_points
