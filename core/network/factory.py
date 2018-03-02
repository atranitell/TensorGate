# -*- coding: utf-8 -*-
""" Contains a factory for building various models.
    Author: Kai JIN
    Updated: 2017-08-28
"""
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from core.network import model

networks_map = {
    'alexnet': model.AlexNet,
    'cifarnet': model.CifarNet,
    'overfeat': model.Overfeat,
    'vgg_a': model.VGG,
    'vgg_16': model.VGG,
    'vgg_19': model.VGG,
    'inception_v1': model.Inception_v1,
    'inception_v2': model.Inception_v2,
    'inception_v3': model.Inception_v3,
    'inception_v4': model.Inception_v4,
    'inception_resnet_v1': model.Inception_Resnet_v1,
    'inception_resnet_v2': model.Inception_Resnet_v2,
    'lenet': model.LeNet,
    'resnet_v1_50': model.ResNet,
    'resnet_v1_101': model.ResNet,
    'resnet_v1_152': model.ResNet,
    'resnet_v1_200': model.ResNet,
    'resnet_v2_50': model.ResNet,
    'resnet_v2_101': model.ResNet,
    'resnet_v2_152': model.ResNet,
    'resnet_v2_200': model.ResNet,
    'mobilenet_v1': model.MobileNet_v1,
    'nasnet_cifar': model.NasNet,
    'nasnet_mobile': model.NasNet,
    'nasnet_large': model.NasNet,
    'squeezenet': model.SqueezeNet,
    'simplenet': model.SimpleNet,
    'mlp': model.MLP,
    'audionet': model.AudioNet
}


def network(X, config, phase, name='', reuse=None):
  """ net factory
  """
  is_training = True if phase == 'train' else False
  with tf.variable_scope(name) as scope:
    net, argscope = networks_map[config.net.name](X, config, is_training)
    if reuse:
      scope.reuse_variables()
    if argscope is not None:
      with arg_scope(argscope):
        return net
    else:
      return net