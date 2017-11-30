# -*- coding: utf-8 -*-
""" Setting task
    Author: Kai JIN
    Updated: 2017-11-23
"""

from config.datasets import mnist
from config.datasets import kinvae
from config.datasets import trafficflow
from config.datasets import celeba

config_map = {
    'mnist': mnist.mnist,
    'mnist.gan': mnist.mnist_gan,
    'mnist.regression': mnist.mnist_regression,
    'kinvae': kinvae.kinvae,
    'kinvae.pair': kinvae.kinvae_pair,
    'trafficflow': trafficflow.trafficflow,
    'celeba': celeba.celeba
}


def get(name):
  """ dataset preconfig factory
  """
  return config_map[name]()
