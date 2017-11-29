# -*- coding: utf-8 -*-
""" Contains a factory for building various models.
    Author: Kai JIN
    Updated: 2017-08-28
"""
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from core.network import cnn

networks_map = {
    'alexnet': cnn.AlexNet,
    'cifarnet': cnn.CifarNet,
    'overfeat': cnn.Overfeat,
    'vgg_a': cnn.VGG,
    'vgg_16': cnn.VGG,
    'vgg_19': cnn.VGG,
    'inception_v1': cnn.Inception_v1,
    'inception_v2': cnn.Inception_v2,
    'inception_v3': cnn.Inception_v3,
    'inception_v4': cnn.Inception_v4,
    'inception_resnet_v2': cnn.Inception_Resnet_v2,
    'lenet': cnn.LeNet,
    'resnet_v1_50': cnn.ResNet,
    'resnet_v1_101': cnn.ResNet,
    'resnet_v1_152': cnn.ResNet,
    'resnet_v1_200': cnn.ResNet,
    'resnet_v2_50': cnn.ResNet,
    'resnet_v2_101': cnn.ResNet,
    'resnet_v2_152': cnn.ResNet,
    'resnet_v2_200': cnn.ResNet,
    'mobilenet_v1': cnn.MobileNet_v1,
    'nasnet_cifar': cnn.NasNet,
    'nasnet_mobile': cnn.NasNet,
    'nasnet_large': cnn.NasNet,
    'lightnet': cnn.LightNet
}


def network(X, config, phase, name='', reuse=None):
  """ net factory
  """
  is_training = True if phase == 'train' else False
  net, argscope = networks_map[config.net.name](X, config, is_training)
  with tf.variable_scope(name) as scope:
    if reuse:
      scope.reuse_variables()
    with arg_scope(argscope):
      return net
