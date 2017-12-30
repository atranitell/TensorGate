# -*- coding: utf-8 -*-
""" Setting task
    Author: Kai JIN
    Updated: 2017-11-23
"""
from config.datasets.mnist import mnist
from config.datasets.mnist import mnist_gan
from config.datasets.mnist import mnist_regression
from config.datasets.kinface import kinvae_pair
from config.datasets.kinface import kinface_npy
from config.datasets.kinface import kinface
from config.datasets.trafficflow import trafficflow
from config.datasets.lfw import lfw

config_map = {
    'mnist': mnist.mnist,
    'mnist.gan': mnist_gan.mnist_gan,
    'mnist.regression': mnist_regression.mnist_regression,
    'kinvae1.pair': kinvae_pair.kinvae1_pair,
    'kinvae2.pair': kinvae_pair.kinvae2_pair,
    'kinface.npy': kinface_npy.kinface_npy,
    'lfw': lfw.lfw,
    'trafficflow': trafficflow.trafficflow,
    'kinface': kinface.Kinface
}


def get(name, extra=None):
  """ dataset preconfig factory
  """
  return config_map[name](extra)
