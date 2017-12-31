# -*- coding: utf-8 -*-
""" All interface to access the framework
    Author: Kai JIN
    Updated: 2017/11/19
"""
import os
import sys
import argparse

# hidden device output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# allocate GPU to sepcify device
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

from config import factory


def run(name, extra):
  config = factory.get(name, extra)

  """Target"""
  # For CNN
  if config.target == 'cnn.classification':
    from issue.cnn.classification import classification as App
  elif config.target == 'cnn.regression':
    from issue.cnn.regression import regression as App
  elif config.target == 'cnn.pairwise':
    from issue.cnn.pairwise import pairwise as App
  # For GAN
  elif config.target == 'gan.dcgan':
    from issue.gan.dcgan import DCGAN as App
  elif config.target == 'gan.cgan':
    from issue.gan.cgan import CGAN as App
  elif config.target == 'gan.cwgan':
    from issue.gan.cwgan import CWGAN as App
  elif config.target == 'gan.acgan':
    from issue.gan.acgan import ACGAN as App
  # For VAE
  elif config.target == 'vae.cvae':
    from issue.vae.cvae import CVAE as App
  elif config.target == 'vae.cvae.gan':
    from issue.vae.cvae_gan import CVAE_GAN as App
  elif config.target.find('kinvae') == 0:
    from issue.kinface.kinface import select as App
  # For ML
  elif config.target == 'ml.active.sampler':
    from issue.ml.active_sampler import active_sampler as App
  elif config.target == 'ml.trafficflow':
    from issue.ml.trafficflow import trafficflow as App
  elif config.target == 'ml.extract_feature':
    from issue.ml.extract_feature import EXTRACT_FEATURE as App
  # Unkown
  else:
    raise ValueError('Unknown target [%s]' % config.target)

  # init
  app = App(config)

  # for task
  if config.task == 'train':
    app.train()
  elif config.task == 'test':
    app.test()
  elif config.task == 'val':
    app.val()
  elif config.task == 'pipline':
    app.pipline()
  else:
    raise ValueError('Unknown task [%s]' % config.task)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-name', type=str, dest='name')
  parser.add_argument('-extra', type=str, dest='extra', default=None)
  args, _ = parser.parse_known_args()
  run(args.name, args.extra)
