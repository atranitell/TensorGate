# -*- coding: utf-8 -*-
""" Setting task
    Author: Kai JIN
    Updated: 2017-11-23
"""
from config.datasets.mnist import mnist
from config.datasets.mnist import mnist_gan
from config.datasets.mnist import mnist_regression
from config.datasets.kinface import kinvae
from config.datasets.kinface import kinvae_pair
from config.datasets.kinface import kinface_npy
from config.datasets.trafficflow import trafficflow

config_map = {
    'mnist': mnist.mnist,
    'mnist.gan': mnist_gan.mnist_gan,
    'mnist.regression': mnist_regression.mnist_regression,
    'kinvae': kinvae.kinvae,
    'kinvae1.pair': kinvae_pair.kinvae1_pair,
    'kinvae2.pair': kinvae_pair.kinvae2_pair,
    'kinface.npy': kinface_npy.kinface_npy,
    'trafficflow': trafficflow.trafficflow
}


def get(name):
  """ dataset preconfig factory
  """
  return config_map[name]()
