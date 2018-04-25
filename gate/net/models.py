# -*- coding: utf-8 -*-
"""
DECTRION FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2018/4/19

--------------------------------------------------------

Models: Re-package the model with standard input parameter.

"""

import tensorflow as tf
arg_scope = tf.contrib.framework.arg_scope

from gate.net.nets import lenet
from gate.net.nets import cifarnet
from gate.net.nets import alexnet
from gate.net.nets import inception
from gate.net.nets import resnet_v1
from gate.net.nets import resnet_v2
from gate.net.nets import overfeat
from gate.net.nets import vgg
from gate.net.nets import mobilenet_v1
from gate.net.nets.mobilenet import mobilenet_v2
from gate.net.nets.nasnet import nasnet

from gate.net.nets import cyclegan
from gate.net.nets import dcgan
from gate.net.nets import pix2pix


def _lenet(X, config, is_training):
  return lenet.lenet(
      X, config.num_classes,
      is_training,
      config.dropout_keep)


def _lenet_scope(config):
  return lenet.lenet_arg_scope(config.weight_decay)


def _cifarnet(X, config, is_training):
  return cifarnet.cifarnet(
      X, num_classes=config.num_classes,
      is_training=is_training,
      dropout_keep_prob=config.dropout_keep)


def _cifarnet_scope(config):
  return cifarnet.cifarnet_arg_scope(config.weight_decay)


def _resnet(X, config, is_training):
  net_fn_map = {
      'resnet_v2_50': resnet_v2.resnet_v2_50,
      'resnet_v2_101': resnet_v2.resnet_v2_101,
      'resnet_v2_152': resnet_v2.resnet_v2_152,
      'resnet_v2_200': resnet_v2.resnet_v2_200,
      'resnet_v1_50': resnet_v1.resnet_v1_50,
      'resnet_v1_101': resnet_v1.resnet_v1_101,
      'resnet_v1_152': resnet_v1.resnet_v1_152,
      'resnet_v1_200': resnet_v1.resnet_v1_200,
  }
  net = net_fn_map[config.name](
      X, num_classes=config.num_classes,
      is_training=is_training,
      global_pool=config.global_pool,
      output_stride=None,
      spatial_squeeze=True)
  return net


def _resnet_scope(config):
  if 'v2' in config.name:
    resnet_scope_fn = resnet_v2.resnet_arg_scope
  else:
    resnet_scope_fn = resnet_v1.resnet_arg_scope
  argscope = resnet_scope_fn(
      weight_decay=config.weight_decay,
      batch_norm_decay=config.batch_norm_decay,
      batch_norm_epsilon=config.batch_norm_epsilon,
      batch_norm_scale=config.batch_norm_scale,
      activation_fn=config.activation_fn,
      use_batch_norm=config.use_batch_norm)
  return argscope
