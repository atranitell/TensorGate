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
sys.path.append('core\\net')
import tensorflow as tf
from tensorflow import layers
from core.net.nets import resnet_v2


def resnet_v2_50_argscope(config):
  return resnet_v2.resnet_arg_scope(
      weight_decay=config.net.weight_decay,
      batch_norm_decay=config.net.batch_norm_decay,
      batch_norm_epsilon=config.net.batch_norm_epsilon,
      batch_norm_scale=config.net.batch_norm_scale,
      activation_fn=config.net.activation_fn,
      use_batch_norm=config.net.use_batch_norm)


def resnet_v2_50(config, inputs, is_training, scope='', reuse=None):
  with tf.variable_scope(scope) as scope:
    argscope = resnet_v2_50_argscope(config)
    with tf.contrib.framework.arg_scope(argscope):
      logit, end_points = resnet_v2.resnet_v2_50(
          inputs=inputs,
          num_classes=config.net.num_classes,
          is_training=is_training,
          global_pool=config.net.global_pool,
          output_stride=config.net.output_stride,
          spatial_squeeze=config.net.spatial_squeeze,
          reuse=reuse,
          scope=config.net.scope)
      return logit, end_points
