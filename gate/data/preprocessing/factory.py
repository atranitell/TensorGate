# -*- coding: utf-8 -*-
"""
GATE FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2017/8/28

--------------------------------------------------------

Data Processing

"""

import tensorflow as tf

from gate.data.preprocessing import cifarnet_preprocessing
from gate.data.preprocessing import avec2014_preprocessing
# from gate.data.preprocessing import inception_preprocessing
# from gate.data.preprocessing import lenet_preprocessing
# from gate.data.preprocessing import vgg_preprocessing

# from gate.data.preprocessing import mnist_gan_preprocessing
# from gate.data.preprocessing import kinship_vae_preprocessing


preprocessing_map = {
    'cifarnet': cifarnet_preprocessing,
    # 'inception': inception_preprocessing,
    # 'lenet': lenet_preprocessing,
    # 'vgg': vgg_preprocessing,
    # 'gan.mnist': mnist_gan_preprocessing,
    # 'vae.kinship': kinship_vae_preprocessing,
    'avec2014': avec2014_preprocessing
}


def get_preprocessing(X, name, cfg, phase):
  """ preprocessing factory
  """
  is_training = True if phase == 'train' else False
  with tf.name_scope('preprocessing/' + name):
    return preprocessing_map[name].preprocess_image(
        X, cfg.output_height, cfg.output_width, is_training)
