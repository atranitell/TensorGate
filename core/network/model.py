# -*- coding: utf-8 -*-
""" Contains a factory for building various models.
    Author: Kai JIN
    Updated: 2017-08-28
"""
import sys
sys.path.append('core/network/')
import tensorflow as tf

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

from core.network.nets import squeezenet
from core.network.nets import inception_resnet_v1

from core.network.custom import simplenet
from core.network.custom import mlp
from core.network.custom import audionet


#------------------------------------------------------------
# SLIM AREA : for tensorflow slim model
#------------------------------------------------------------


def _cifarnet(X, config, is_train):
  return cifarnet.cifarnet(
      X, num_classes=config.data.num_classes,
      is_training=is_train,
      dropout_keep_prob=config.net.dropout_keep)


def _cifarnet_scope(config):
  return cifarnet.cifarnet_arg_scope(config.net.weight_decay)


def _resnet(X, config, is_train):
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
  net = net_fn_map[config.net.name](
      X, num_classes=config.data.num_classes,
      is_training=is_train,
      global_pool=True,
      output_stride=None,
      spatial_squeeze=True)
  return net


def _resnet_scope(config):
  if 'v2' in config.net.name:
    resnet_scope_fn = resnet_v2.resnet_arg_scope
  else:
    resnet_scope_fn = resnet_v1.resnet_arg_scope
  argscope = resnet_scope_fn(
      weight_decay=config.net.weight_decay,
      batch_norm_decay=config.net.batch_norm_decay,
      batch_norm_epsilon=config.net.batch_norm_epsilon,
      batch_norm_scale=config.net.batch_norm_scale)
  return argscope


def _alexnet(X, config, is_train):
  return alexnet.alexnet_v2(
      X, num_classes=config.data.num_classes,
      is_training=is_train,
      dropout_keep_prob=config.net.dropout_keep)


def _alexnet_scope(config):
  return alexnet.alexnet_v2_arg_scope(config.net.weight_decay)


def _inception_scope(config):
  arg_scope_fn = {
      'inception_v1': inception.inception_v1_arg_scope,
      'inception_v2': inception.inception_v2_arg_scope,
      'inception_v3': inception.inception_v3_arg_scope,
      'inception_v4': inception.inception_v4_arg_scope
  }
  return arg_scope_fn[config.net.name](
      weight_decay=config.net.weight_decay,
      use_batch_norm=config.net.use_batch_norm,
      batch_norm_decay=config.net.batch_norm_decay,
      batch_norm_epsilon=config.net.batch_norm_epsilon,
      activation_fn=config.net.activation_fn)


def _inception_v1(X, config, is_train):
  return inception.inception_v1(
      X, num_classes=config.data.num_classes,
      is_training=is_train,
      dropout_keep_prob=config.net.dropout_keep,
      spatial_squeeze=True,
      global_pool=False)


def _inception_v2(X, config, is_train):
  return inception.inception_v2(
      X, num_classes=config.data.num_classes,
      is_training=is_train,
      dropout_keep_prob=config.net.dropout_keep,
      min_depth=16,
      depth_multiplier=1.0,
      spatial_squeeze=True,
      global_pool=False)


def _inception_v3(X, config, is_train):
  return inception.inception_v3(
      X, num_classes=config.data.num_classes,
      is_training=is_train,
      dropout_keep_prob=config.net.dropout_keep,
      min_depth=16,
      depth_multiplier=1.0,
      spatial_squeeze=True,
      global_pool=False)


def _inception_v4(X, config, is_train):
  return inception.inception_v4(
      X, num_classes=config.data.num_classes + 1,
      is_training=is_train,
      dropout_keep_prob=config.net.dropout_keep,
      create_aux_logits=True)


def _inception_resnet_v2_scope(config):
  return inception.inception_resnet_v2_arg_scope(
      weight_decay=config.net.weight_decay,
      batch_norm_decay=config.net.batch_norm_decay,
      batch_norm_epsilon=config.net.batch_norm_epsilon,
      activation_fn=config.net.activation_fn)


def _inception_resnet_v2(X, config, is_train):
  return inception.inception_resnet_v2(
      X, num_classes=config.data.num_classes + 1,
      is_training=is_train,
      dropout_keep_prob=config.net.dropout_keep,
      create_aux_logits=True,
      activation_fn=config.net.activation_fn)


def _vgg_scope(config):
  return vgg.vgg_arg_scope(config.net.weight_decay)


def _vgg(X, config, is_train):
  net_fn_map = {
      'vgg_a': vgg.vgg_a,
      'vgg_16': vgg.vgg_16,
      'vgg_19': vgg.vgg_19,
      'vgg_d': vgg.vgg_d,
      'vgg_e': vgg.vgg_e
  }
  return net_fn_map[config.net.name](
      X, num_classes=config.data.num_classes,
      is_training=is_train,
      dropout_keep_prob=config.net.dropout_keep,
      spatial_squeeze=True,
      fc_conv_padding='VALID',
      global_pool=False)


def _lenet_scope(config):
  return lenet.lenet_arg_scope(config.net.weight_decay)


def _lenet(X, config, is_train):
  return lenet.lenet(
      X, num_classes=config.data.num_classes,
      is_training=is_train,
      dropout_keep_prob=config.net.dropout_keep,
      prediction_fn=tf.nn.softmax)


def _overfeat_scope(config):
  return overfeat.overfeat_arg_scope(config.net.weight_decay)


def _overfeat(X, config, is_train):
  return overfeat.overfeat(
      X, num_classes=config.data.num_classes,
      is_training=is_train,
      dropout_keep_prob=config.net.dropout_keep,
      spatial_squeeze=True,
      global_pool=False)


def _mobilenet_v1_scope(config):
  return mobilenet_v1.mobilenet_v1_arg_scope(
      is_training=is_train,
      weight_decay=0.00004,
      stddev=0.09,
      regularize_depthwise=False)


def _mobilenet_v1(X, config, is_train):
  return mobilenet_v1.mobilenet_v1(
      X, num_classes=config.net.num_classes,
      dropout_keep_prob=config.net.dropout_keep,
      is_training=is_train,
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
  return scope_fn_map[config.net.name](
      weight_decay=config.net.weight_decay,
      batch_norm_decay=config.net.batch_norm_decay,
      batch_norm_epsilon=config.net.batch_norm_epsilon)


def _nasnet(X, config, is_train):
  net_fn_map = {
      'nasnet_cifar': nasnet.build_nasnet_cifar,
      'nasnet_mobile': nasnet.build_nasnet_mobile,
      'nasnet_large': nasnet.build_nasnet_large
  }
  return net_fn_map[config.net.name](X, config.data.num_classes, is_train)

#------------------------------------------------------------
# PUBLIC AREA : for public model
#------------------------------------------------------------


def _squeezenet(X, config, is_train):
  return squeezenet.inference(X, config.net.dropout_keep, is_train)


def _inception_resnet_v1(X, config, is_train):
  return inception_resnet_v1.inception_resnet_v1(
      X, is_train, config.net.dropout_keep)

#------------------------------------------------------------
# CUSTOM AREA : for own network model
#------------------------------------------------------------


def _simplenet(X, config, is_train):
  return simplenet.simplenet(
      X, num_classes=config.data.num_classes,
      is_training=is_train)


def _mlp(X, config, is_train):
  return mlp.mlp(X, num_classes=config.data.num_classes)


def _audionet(X, config, is_train):
  model = audionet.AudioNet()
  return model.model(X, config.data.num_classes, is_train)
