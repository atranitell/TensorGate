# -*- coding: utf-8 -*-
""" Setting task
    Author: Kai JIN
    Updated: 2017-11-23
"""
from config.datasets.mnist import mnist
from config.datasets.mnist import mnist_gan
from config.datasets.mnist import mnist_regression
from config.datasets.kinface import kinface
from config.datasets.kinface import kinvae
from config.datasets.kinface import kinvae_pair
from config.datasets.celeba import celeba
from config.datasets.trafficflow import trafficflow

config_map = {
    'mnist': mnist.mnist,
    'mnist.gan': mnist_gan.mnist_gan,
    'mnist.regression': mnist_regression.mnist_regression,
    'kinvae': kinvae.kinvae,
    'kinvae.pair': kinvae_pair.kinvae_pair,
    'kinface': kinface.kinface,
    'trafficflow': trafficflow.trafficflow,
    'celeba': celeba.celeba
}


def get(name):
  """ dataset preconfig factory
  """
  return config_map[name]()
