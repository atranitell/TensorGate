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
from core.network.cnns import lightnet


def CifarNet(X, config, is_train):
  """ CifarNet """
  argscope = cifarnet.cifarnet_arg_scope(
      config.net.weight_decay)
  net = cifarnet.cifarnet(
      X, num_classes=config.data.num_classes,
      is_training=is_train,
      dropout_keep_prob=config.net.dropout_keep)
  return net, argscope


def ResNet(X, config, is_train):
  """ ResNet v1/v2 """
  if 'v2' in config.net.name:
    resnet_scope_fn = resnet_v2.resnet_arg_scope
  else:
    resnet_scope_fn = resnet_v1.resnet_arg_scope
  argscope = resnet_scope_fn(
      weight_decay=config.net.weight_decay,
      batch_norm_decay=config.net.batch_norm_decay,
      batch_norm_epsilon=config.net.batch_norm_epsilon,
      batch_norm_scale=config.net.batch_norm_scale,
      activation_fn=config.net.activation_fn,
      use_batch_norm=config.net.use_batch_norm)
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
  return net, argscope


def AlexNet(X, config, is_train):
  """ AlexNet """
  argscope = alexnet.alexnet_v2_arg_scope(
      config.net.weight_decay)
  net = alexnet.alexnet_v2(
      X, num_classes=config.data.num_classes,
      is_training=is_train,
      dropout_keep_prob=config.net.dropout_keep)
  return net, argscope


def Inception_scope(config, argscope_fn):
  return argscope_fn(
      weight_decay=config.net.weight_decay,
      use_batch_norm=config.net.use_batch_norm,
      batch_norm_decay=config.net.batch_norm_decay,
      batch_norm_epsilon=config.net.batch_norm_epsilon,
      activation_fn=config.net.activation_fn)


def Inception_v1(X, config, is_train):
  argscope = Inception_scope(config, inception.inception_v1_arg_scope)
  net = inception.inception_v1(
      X, num_classes=config.data.num_classes,
      is_training=is_train,
      dropout_keep_prob=config.net.dropout_keep,
      prediction_fn=tf.nn.softmax,
      spatial_squeeze=True,
      global_pool=False)
  return net, argscope


def Inception_v2(X, config, is_train):
  argscope = Inception_scope(config, inception.inception_v2_arg_scope)
  net = inception.inception_v2(
      X, num_classes=config.data.num_classes,
      is_training=is_train,
      dropout_keep_prob=config.net.dropout_keep,
      min_depth=16,
      depth_multiplier=1.0,
      prediction_fn=tf.nn.softmax,
      spatial_squeeze=True,
      global_pool=False)
  return net, argscope


def Inception_v3(X, config, is_train):
  argscope = Inception_scope(config, inception.inception_v3_arg_scope)
  net = inception.inception_v3(
      X, num_classes=config.data.num_classes,
      is_training=is_train,
      dropout_keep_prob=config.net.dropout_keep,
      min_depth=16,
      depth_multiplier=1.0,
      prediction_fn=tf.nn.softmax,
      spatial_squeeze=True,
      global_pool=False)
  return net, argscope


def Inception_v4(X, config, is_train):
  argscope = Inception_scope(config, inception.inception_v4_arg_scope)
  net = inception.inception_v4(
      X, num_classes=config.data.num_classes + 1,
      is_training=is_train,
      dropout_keep_prob=config.net.dropout_keep,
      create_aux_logits=True)
  return net, argscope


def Inception_Resnet_v2(X, config, is_train):
  argscope = inception.inception_resnet_v2_arg_scope(
      weight_decay=config.net.weight_decay,
      batch_norm_decay=config.net.batch_norm_decay,
      batch_norm_epsilon=config.net.batch_norm_epsilon,
      activation_fn=config.net.activation_fn)
  net = inception.inception_resnet_v2(
      X, num_classes=config.data.num_classes + 1,
      is_training=is_train,
      dropout_keep_prob=config.net.dropout_keep,
      create_aux_logits=True,
      activation_fn=config.net.activation_fn)
  return net, argscope


def VGG(X, config, is_train):
  argscope = vgg.vgg_arg_scope(config.net.weight_decay)
  net_fn_map = {
      'vgg_a': vgg.vgg_a,
      'vgg_16': vgg.vgg_16,
      'vgg_19': vgg.vgg_19,
      'vgg_d': vgg.vgg_d,
      'vgg_e': vgg.vgg_e
  }
  net = net_fn_map[config.net.name](
      X, num_classes=config.data.num_classes,
      is_training=is_train,
      dropout_keep_prob=config.net.dropout_keep,
      spatial_squeeze=True,
      fc_conv_padding='VALID',
      global_pool=False)
  return net, argscope


def LeNet(X, config, is_train):
  argscope = lenet.lenet_arg_scope(config.net.weight_decay)
  net = lenet.lenet(
      X, num_classes=config.data.num_classes,
      is_training=is_train,
      dropout_keep_prob=config.net.dropout_keep,
      prediction_fn=tf.nn.softmax)
  return net, argscope


def Overfeat(X, config, is_train):
  argscope = overfeat.overfeat_arg_scope(config.net.weight_decay)
  net = overfeat.overfeat(
      X, num_classes=config.data.num_classes,
      is_training=is_train,
      dropout_keep_prob=config.net.dropout_keep,
      spatial_squeeze=True,
      global_pool=False)
  return net, argscope


def MobileNet_v1(X, config, is_train):
  argscope = mobilenet_v1.mobilenet_v1_arg_scope(
      is_training=is_train,
      weight_decay=0.00004,
      stddev=0.09,
      regularize_depthwise=False)
  net = mobilenet_v1.mobilenet_v1(
      X, num_classes=config.net.num_classes,
      dropout_keep_prob=config.net.dropout_keep,
      is_training=is_train,
      min_depth=8,
      depth_multiplier=1.0,
      conv_defs=None,
      prediction_fn=tf.contrib.layers.softmax,
      spatial_squeeze=True,
      global_pool=False)
  return net, argscope


def NasNet(X, config, is_train):
  scope_fn_map = {
      'nasnet_cifar': nasnet.nasnet_cifar_arg_scope,
      'nasnet_mobile': nasnet.nasnet_mobile_arg_scope,
      'nasnet_large': nasnet.nasnet_mobile_arg_scope
  }
  net_fn_map = {
      'nasnet_cifar': nasnet.build_nasnet_cifar,
      'nasnet_mobile': nasnet.build_nasnet_mobile,
      'nasnet_large': nasnet.build_nasnet_large
  }
  argscope = scope_fn_map[config.net.name](
      weight_decay=config.net.weight_decay,
      batch_norm_decay=config.net.batch_norm_decay,
      batch_norm_epsilon=config.net.batch_norm_epsilon)
  net = net_fn_map[config.net.name](X, config.data.num_classes, is_train)
  return net, argscope


def LightNet(X, config, is_train):
  argscope = lightnet.lightnet_argscope(
      weight_decay=config.net.weight_decay,
      batch_norm_decay=config.net.batch_norm_decay,
      batch_norm_epsilon=config.net.batch_norm_epsilon,
      batch_norm_scale=config.net.batch_norm_scale)
  net = lightnet.lightnet(
      X, num_classes=config.data.num_classes,
      is_training=is_train,
      dropout_keep_prob=config.net.dropout_keep)
  return net, argscope
