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

from gate.net.custom import audionet

from gate.net.deepfuse import resnet_v2_bishared
from gate.net.deepfuse import vgg_bishared
from gate.net.deepfuse import alexnet_bishared

# -------------------------------------------------------
# SLIM
# -------------------------------------------------------


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


def _alexnet(X, config, is_training):
  return alexnet.alexnet_v2(
      inputs=X,
      num_classes=config.num_classes,
      is_training=is_training,
      dropout_keep_prob=config.dropout_keep,
      spatial_squeeze=config.spatial_squeeze,
      scope=config.scope,
      global_pool=config.global_pool)


def _alexnet_scope(config):
  return alexnet.alexnet_v2_arg_scope(config.weight_decay)


def _inception_scope(config):
  arg_scope_fn = {
      'inception_v1': inception.inception_v1_arg_scope,
      'inception_v2': inception.inception_v2_arg_scope,
      'inception_v3': inception.inception_v3_arg_scope,
      'inception_v4': inception.inception_v4_arg_scope
  }
  return arg_scope_fn[config.name](
      weight_decay=config.weight_decay,
      use_batch_norm=config.use_batch_norm,
      batch_norm_decay=config.batch_norm_decay,
      batch_norm_epsilon=config.batch_norm_epsilon,
      activation_fn=config.activation_fn)


def _inception_v1(X, config, is_training):
  return inception.inception_v1(
      X, num_classes=config.num_classes,
      is_training=is_training,
      dropout_keep_prob=config.dropout_keep,
      spatial_squeeze=True,
      global_pool=False)


def _inception_v2(X, config, is_training):
  return inception.inception_v2(
      X, num_classes=config.num_classes,
      is_training=is_training,
      dropout_keep_prob=config.dropout_keep,
      min_depth=16,
      depth_multiplier=1.0,
      spatial_squeeze=True,
      global_pool=False)


def _inception_v3(X, config, is_training):
  return inception.inception_v3(
      X, num_classes=config.num_classes,
      is_training=is_training,
      dropout_keep_prob=config.dropout_keep,
      min_depth=16,
      depth_multiplier=1.0,
      spatial_squeeze=True,
      global_pool=False)


def _inception_v4(X, config, is_training):
  return inception.inception_v4(
      X, num_classes=config.num_classes,
      is_training=is_training,
      dropout_keep_prob=config.dropout_keep,
      create_aux_logits=True)


def _inception_resnet_v2_scope(config):
  return inception.inception_resnet_v2_arg_scope(
      weight_decay=config.weight_decay,
      batch_norm_decay=config.batch_norm_decay,
      batch_norm_epsilon=config.batch_norm_epsilon,
      activation_fn=config.activation_fn)


def _inception_resnet_v2(X, config, is_training):
  return inception.inception_resnet_v2(
      X, num_classes=config.num_classes,
      is_training=is_training,
      dropout_keep_prob=config.dropout_keep,
      create_aux_logits=True,
      activation_fn=config.activation_fn)


def _vgg_scope(config):
  return vgg.vgg_arg_scope(config.weight_decay)


def _vgg(X, config, is_training):
  net_fn_map = {
      'vgg_11': vgg.vgg_a,
      'vgg_16': vgg.vgg_16,
      'vgg_19': vgg.vgg_19,
  }
  return net_fn_map[config.name](
      X, num_classes=config.num_classes,
      is_training=is_training,
      dropout_keep_prob=config.dropout_keep,
      spatial_squeeze=True,
      fc_conv_padding='VALID',
      global_pool=False)


def _overfeat_scope(config):
  return overfeat.overfeat_arg_scope(config.weight_decay)


def _overfeat(X, config, is_training):
  return overfeat.overfeat(
      X, num_classes=config.num_classes,
      is_training=is_training,
      dropout_keep_prob=config.dropout_keep,
      spatial_squeeze=True,
      global_pool=False)


def _mobilenet_v1_scope(config, is_training):
  return mobilenet_v1.mobilenet_v1_arg_scope(
      is_training=is_training,
      weight_decay=0.00004,
      stddev=0.09,
      regularize_depthwise=False)


def _mobilenet_v1(X, config, is_training):
  return mobilenet_v1.mobilenet_v1(
      X, num_classes=config.num_classes,
      dropout_keep_prob=config.dropout_keep,
      is_training=is_training,
      min_depth=8,
      depth_multiplier=1.0,
      conv_defs=None,
      prediction_fn=tf.contrib.layers.softmax,
      spatial_squeeze=True,
      global_pool=False)


def _nasnet_scope(config):
  scope_fn_map = {
      'nasnet_cifar': nasnet.nasnet_cifar_arg_scope,
      'nasnet_mobile': nasnet.nasnet_mobile_arg_scope,
      'nasnet_large': nasnet.nasnet_mobile_arg_scope
  }
  return scope_fn_map[config.name](
      weight_decay=config.weight_decay,
      batch_norm_decay=config.batch_norm_decay,
      batch_norm_epsilon=config.batch_norm_epsilon)


def _nasnet(X, config, is_training):
  net_fn_map = {
      'nasnet_cifar': nasnet.build_nasnet_cifar,
      'nasnet_mobile': nasnet.build_nasnet_mobile,
      'nasnet_large': nasnet.build_nasnet_large
  }
  return net_fn_map[config.name](X, config.num_classes, is_training)


# -------------------------------------------------------
# custom
# -------------------------------------------------------


def _audionet(X, config, is_training):
  model = audionet.AudioNet()
  return model.model(X, config.num_classes, is_training)


# -------------------------------------------------------
# DEEP Fuse
# -------------------------------------------------------


def _bisahred_resnet(X, config, is_train):
  net_fn_map = {
      'resnet_v2_50_bishared': resnet_v2_bishared.resnet_v2_50,
      'resnet_v2_101_bishared': resnet_v2_bishared.resnet_v2_101,
      'resnet_v2_152_bishared': resnet_v2_bishared.resnet_v2_152,
      'resnet_v2_200_bishared': resnet_v2_bishared.resnet_v2_200
  }
  net = net_fn_map[config.name](
      X, num_classes=config.num_classes,
      is_training=is_train,
      global_pool=True,
      output_stride=None,
      spatial_squeeze=True)
  return net


def _bisahred_resnet_scope(config):
  resnet_scope_fn = resnet_v2_bishared.resnet_arg_scope
  argscope = resnet_scope_fn(
      weight_decay=config.weight_decay,
      batch_norm_decay=config.batch_norm_decay,
      batch_norm_epsilon=config.batch_norm_epsilon,
      batch_norm_scale=config.batch_norm_scale)
  return argscope


def _vgg_bishared_scope(config):
  return vgg_bishared.vgg_arg_scope(config.weight_decay)


def _vgg_bishared(X, config, is_training):
  net_fn_map = {
      'vgg_11_bishared': vgg_bishared.vgg_a,
      'vgg_16_bishared': vgg_bishared.vgg_16,
      'vgg_19_bishared': vgg_bishared.vgg_19,
  }
  return net_fn_map[config.name](
      X, num_classes=config.num_classes,
      is_training=is_training,
      dropout_keep_prob=config.dropout_keep,
      spatial_squeeze=config.spatial_squeeze,
      fc_conv_padding='VALID',
      global_pool=config.global_pool)


def _alexnet_bishared(X, config, is_training):
  return alexnet_bishared.alexnet_v2(
      inputs=X,
      num_classes=config.num_classes,
      is_training=is_training,
      dropout_keep_prob=config.dropout_keep,
      spatial_squeeze=config.spatial_squeeze,
      scope=config.name,
      global_pool=config.global_pool)


def _alexnet_bishared_scope(config):
  return alexnet_bishared.alexnet_v2_arg_scope(config.weight_decay)
