# Copyright 2017 The KaiJIN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""NETWORK FACTORY: Parse parameters from the config"""

import tensorflow as tf
from gate.net import net_models

arg_scope = tf.contrib.framework.arg_scope

network_map = {
    'alexnet_v2': net_models._alexnet,
    'cifarnet': net_models._cifarnet,
    'overfeat': net_models._overfeat,
    'vgg_11': net_models._vgg,
    'vgg_16': net_models._vgg,
    'vgg_19': net_models._vgg,
    'inception_v1': net_models._inception_v1,
    'inception_v2': net_models._inception_v2,
    'inception_v3': net_models._inception_v3,
    'inception_v4': net_models._inception_v4,
    # 'inception_resnet_v1': net_models._inception_resnet_v1,
    'inception_resnet_v2': net_models._inception_resnet_v2,
    'lenet': net_models._lenet,
    'resnet_v1_50': net_models._resnet,
    'resnet_v1_101': net_models._resnet,
    'resnet_v1_152': net_models._resnet,
    'resnet_v1_200': net_models._resnet,
    'resnet_v2_50': net_models._resnet,
    'resnet_v2_101': net_models._resnet,
    'resnet_v2_152': net_models._resnet,
    'resnet_v2_200': net_models._resnet,
    'mobilenet_v1': net_models._mobilenet_v1,
    'nasnet_cifar': net_models._nasnet,
    'nasnet_mobile': net_models._nasnet,
    'nasnet_large': net_models._nasnet,
    # 'squeezenet': model._squeezenet,
    # 'simplenet': model._simplenet,
    # 'mlp': model._mlp,
    'audionet': net_models._audionet,
    'resnet_v2_50_bishared': net_models._bisahred_resnet,
    'resnet_v2_101_bishared': net_models._bisahred_resnet,
    'resnet_v2_152_bishared': net_models._bisahred_resnet,
    'resnet_v2_200_bishared': net_models._bisahred_resnet,
    'vgg_11_bishared': net_models._vgg_bishared,
    'vgg_16_bishared': net_models._vgg_bishared,
    'vgg_19_bishared': net_models._vgg_bishared,
    'alexnet_v2_bishared': net_models._alexnet_bishared,
    'resnet_v2_critical_50': net_models._critical_resnet
}

argscope_map = {
    'alexnet_v2': net_models._alexnet_scope,
    'cifarnet': net_models._cifarnet_scope,
    'overfeat': net_models._overfeat_scope,
    'vgg_11': net_models._vgg_scope,
    'vgg_16': net_models._vgg_scope,
    'vgg_19': net_models._vgg_scope,
    'inception_v1': net_models._inception_scope,
    'inception_v2': net_models._inception_scope,
    'inception_v3': net_models._inception_scope,
    'inception_v4': net_models._inception_scope,
    # 'inception_resnet_v1': None,
    'inception_resnet_v2': net_models._inception_resnet_v2_scope,
    'lenet': net_models._lenet_scope,
    'resnet_v1_50': net_models._resnet_scope,
    'resnet_v1_101': net_models._resnet_scope,
    'resnet_v1_152': net_models._resnet_scope,
    'resnet_v1_200': net_models._resnet_scope,
    'resnet_v2_50': net_models._resnet_scope,
    'resnet_v2_101': net_models._resnet_scope,
    'resnet_v2_152': net_models._resnet_scope,
    'resnet_v2_200': net_models._resnet_scope,
    'mobilenet_v1': net_models._mobilenet_v1_scope,
    'nasnet_cifar': net_models._nasnet_scope,
    'nasnet_mobile': net_models._nasnet_scope,
    'nasnet_large': net_models._nasnet_scope,
    # 'squeezenet': None,
    # 'simplenet': None,
    # 'mlp': None,
    'audionet': None,
    'resnet_v2_50_bishared': net_models._bisahred_resnet_scope,
    'resnet_v2_101_bishared': net_models._bisahred_resnet_scope,
    'resnet_v2_152_bishared': net_models._bisahred_resnet_scope,
    'resnet_v2_200_bishared': net_models._bisahred_resnet_scope,
    'vgg_11_bishared': net_models._vgg_bishared_scope,
    'vgg_16_bishared': net_models._vgg_bishared_scope,
    'vgg_19_bishared': net_models._vgg_bishared_scope,
    'alexnet_v2_bishared': net_models._alexnet_bishared_scope,
    'resnet_v2_critical_50': net_models._critical_resnet_scope
}


def net_graph(X, config, phase, name='', reuse=None):
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
