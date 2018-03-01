# -*- coding: utf-8 -*-
""" Setting task
    Author: Kai JIN
    Updated: 2017-11-23
"""
from config.dataset import mnist
from config.dataset import kinface
from config.dataset import lfw
from config.dataset import trafficflow
from config.dataset import avec2014

config_map = {
    'mnist': mnist.MNIST,
    'mnist.gan': mnist.MNISTGAN,
    'mnist.regression': mnist.MNISTRegression,
    'kinface.vae': kinface.KinfaceVAE,
    'kinface.npy': kinface.KinfaceNPY,
    'lfw': lfw.LFW,
    'trafficflow': trafficflow.TrafficFlow,
    'avec2014_audio': avec2014.AVEC2014_AUDIO
}


def get(name, extra=None):
  """ dataset preconfig factory
  """
  return config_map[name](extra)
