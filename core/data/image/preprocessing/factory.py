# -*- coding: utf-8 -*-
""" Preprocessing Factory.
    Author: Kai JIN
    Updated: 2017-08-28
"""
import tensorflow as tf

from core.data.image.preprocessing import cifarnet_preprocessing
from core.data.image.preprocessing import inception_preprocessing
from core.data.image.preprocessing import lenet_preprocessing
from core.data.image.preprocessing import vgg_preprocessing

from core.data.image.preprocessing import mnist_gan_preprocessing
from core.data.image.preprocessing import kinship_vae_preprocessing


preprocessing_map = {
    'cifarnet': cifarnet_preprocessing,
    'inception': inception_preprocessing,
    'lenet': lenet_preprocessing,
    'vgg': vgg_preprocessing,
    'gan.mnist': mnist_gan_preprocessing,
    'vae.kinship': kinship_vae_preprocessing,
}


def preprocessing(X, name, cfg, phase):
  """ preprocessing factory
  """
  is_training = True if phase == 'train' else False
  with tf.name_scope('preprocessing/' + name):
    return preprocessing_map[name].preprocess_image(
        X, cfg.output_height, cfg.output_width, is_training)
