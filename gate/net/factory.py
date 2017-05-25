# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a factory for building various models."""

import tensorflow as tf

from gate.net import alexnet
from gate.net import cifarnet
from gate.net import inception
from gate.net import lenet
from gate.net import overfeat
from gate.net import resnet_v1
from gate.net import resnet_v2
from gate.net import vgg
from gate.net import lightnet
from gate.net import mlp

slim = tf.contrib.slim

networks_map = {
    'alexnet_v2': alexnet.Alexnet_v2,
    'cifarnet': cifarnet.Cifarnet,
    'overfeat': overfeat.Overfeat,
    'vgg_a': vgg.Vgg_a,
    'vgg_16': vgg.Vgg_16,
    'vgg_19': vgg.Vgg_19,
    'inception_v1': inception.Inception_v1,
    'inception_v2': inception.Inception_v2,
    'inception_v3': inception.Inception_v3,
    'inception_v4': inception.Inception_v4,
    'inception_resnet_v1': inception.Inception_resnet_v1,
    'inception_resnet_v2': inception.Inception_resnet_v2,
    'lenet': lenet.Lenet,
    'resnet_v1_50': resnet_v1.Resnet_v1_50,
    'resnet_v1_101': resnet_v1.Resnet_v1_101,
    'resnet_v1_152': resnet_v1.Resnet_v1_152,
    'resnet_v1_200': resnet_v1.Resnet_v1_200,
    'resnet_v2_50': resnet_v2.Resnet_v2_50,
    'resnet_v2_101': resnet_v2.Resnet_v2_101,
    'resnet_v2_152': resnet_v2.Resnet_v2_152,
    'resnet_v2_200': resnet_v2.Resnet_v2_200,
    'lightnet': lightnet.Lightnet_bn,
    'mlp': mlp.MLP,
}


def check_network(name, data_type):
    """ check network name """
    if name not in networks_map:
        raise ValueError('Unknown network name %s' % name)
    if data_type == 'train':
        return True
    elif data_type == 'val':
        return False
    elif data_type == 'test':
        return False


def get_network(hps, data_type, images, num_classes, name_scope='', reuse=None):
    """Returns a network_fn such as `logits, end_points = network_fn(images)`.

    Args:
      name: The name of the network.
      num_classes: The number of classes to use for classification.
      weight_decay: The l2 coefficient for the model weights.
      is_training: `True` if the model is being used for training and `False`
        otherwise.

    Returns:
      network_fn: A function that applies the model to a batch of images. It has
        the following signature:
          logits, end_points = network_fn(images)
    Raises:
      ValueError: If network `name` is not recognized.
    """
    is_training = check_network(hps.net_name, data_type)
    net = networks_map[hps.net_name](hps, reuse)
    with tf.variable_scope(name_scope) as scope:
        if reuse:
            scope.reuse_variables()
        with slim.arg_scope(net.arguments_scope()):
            return net.model(images, num_classes, is_training)
