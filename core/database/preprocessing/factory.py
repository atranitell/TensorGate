# -*- coding: utf-8 -*-
""" Preprocessing Factory.
    Author: Kai JIN
    Updated: 2017-08-28
"""
import tensorflow as tf

from core.database.preprocessing.slim import cifarnet_preprocessing
from core.database.preprocessing.slim import inception_preprocessing
from core.database.preprocessing.slim import lenet_preprocessing
from core.database.preprocessing.slim import vgg_preprocessing

from core.database.preprocessing.gan import mnist_preprocessing
from core.database.preprocessing.vae import kinship_preprocessing


def is_training(phase):
  return True if phase == 'train' else False


def cifarnet_fn(X, phase, cfg):
  """ preprocessing cifarnet
  """
  return cifarnet_preprocessing.preprocess_image(
      X, cfg.output_height, cfg.output_width, is_training(phase))


def inception_fn(X, phase, cfg):
  """ preprocessing inception
  """
  return inception_preprocessing.preprocess_image(
      X, cfg.output_height, cfg.output_width, is_training(phase),
      bbox=None, fast_mode=True)


def lenet_fn(X, phase, cfg):
  """ preprocessing lenet
  """
  return lenet_preprocessing.preprocess_image(
      X, cfg.output_height, cfg.output_width, is_training(phase))


def vgg_fn(X, phase, cfg):
  """ preprocessing vgg
  """
  return vgg_preprocessing.preprocess_image(
      X, cfg.output_height, cfg.output_width, is_training(phase))


def gan_mnist_fn(X, phase, cfg):
  """ mnist like image (3channels) for gan
  """
  return mnist_preprocessing.preprocess_image(
      X, cfg.output_height, cfg.output_width, is_training(phase))


def vae_kinship_fn(X, phase, cfg):
  return kinship_preprocessing.preprocess_image(
      X, cfg.output_height, cfg.output_width, is_training(phase))


preprocessing_map = {
    'cifarnet': cifarnet_fn,
    'inception': inception_fn,
    'lenet': lenet_fn,
    'vgg': vgg_fn,
    'gan.mnist': gan_mnist_fn,
    'vae.kinship': vae_kinship_fn
}


def preprocessing(name):
  with tf.name_scope('preprocessing/' + name):
    return preprocessing_map[name]
