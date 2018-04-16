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
from config.dataset import imagenet

config_map = {
    'mnist': mnist.MNIST,
    'mnist.gan': mnist.MNISTGAN,
    'mnist.regression': mnist.MNISTRegression,
    'kinface.vae': kinface.KinfaceVAE,
    'kinface.npy': kinface.KinfaceNPY,
    'lfw': lfw.LFW,
    'trafficflow': trafficflow.TrafficFlow,
    'avec2014': avec2014.AVEC2014,
    'avec2014.flow': avec2014.AVEC2014_FLOW,
    'avec2014.audio': avec2014.AVEC2014_AUDIO,
    'avec2014.bi': avec2014.AVEC2014_BI,
    'avec2014.bishared': avec2014.AVEC2014_BISHARED,
    'imagenet': imagenet.ImageNet
}


def get(name, extra=None):
  """ dataset preconfig factory
  """
  return config_map[name](extra)
