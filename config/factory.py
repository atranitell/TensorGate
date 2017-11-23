# -*- coding: utf-8 -*-
""" Setting task
    Author: Kai JIN
    Updated: 2017-11-23
"""

from config.mnist import mnist
from config.mnist import mnist_gan

config_map = {
    'mnist': mnist.mnist,
    'mnist_gan': mnist_gan.mnist_gan
}


def get(name):
  """ dataset preconfig factory
  """
  return config_map[name]()
