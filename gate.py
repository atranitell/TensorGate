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
    from example.cnn.classification import classification as App
  elif config.target == 'cnn.regression':
    from example.cnn.regression import regression as App
  # For functions
  elif config.target == 'extract_feature':
    from example.extract_feature import EXTRACT_FEATURE as App
  # For GAN
  elif config.target == 'gan.dcgan':
    from example.gan.dcgan import DCGAN as App
  elif config.target == 'gan.cgan':
    from example.gan.cgan import CGAN as App
  elif config.target == 'gan.cwgan':
    from example.gan.cwgan import CWGAN as App
  elif config.target == 'gan.acgan':
    from example.gan.acgan import ACGAN as App
  # For ISSUE
  elif config.target.find('kinvae') == 0:
    from issue.kinface.kinface import select as App
  elif config.target == 'trafficflow':
    from issue.trafficnet import trafficflow as App
  elif config.target.find('avec') == 0:
    from issue.avec.avec import select as App
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
  elif config.task == 'heatmap':
    app.heatmap()
  else:
    raise ValueError('Unknown task [%s]' % config.task)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-name', type=str, dest='name')
  parser.add_argument('-extra', type=str, dest='extra', default=None)
  args, _ = parser.parse_known_args()
  run(args.name, args.extra)
