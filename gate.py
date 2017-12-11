# -*- coding: utf-8 -*-
""" All interface to access the framework
    Author: Kai JIN
    Updated: 2017/11/19
"""
import os
import sys
import argparse
from datetime import datetime

# hidden device output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# allocate GPU to sepcify device
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

from config import factory
from core.utils import filesystem
from core.utils.logger import logger

OUTPUTS = filesystem.mkdir('../_outputs/')


class Gate():

  def __init__(self):
    """
    """
    self.config = None
    self.pid = datetime.strftime(datetime.now(), '%y%m%d%H%M%S')

  def initilize(self, name):
    # 0. load config params, default of 'train'
    config = factory.get(name)

    # 1. setting output dir
    if config.output_dir is None:
      config.output_dir = filesystem.mkdir(
          OUTPUTS + config.name + '.' + config.target + '.' + self.pid)

    # 2. setting logger location
    logger.init(config.name + '.' + self.pid, config.output_dir)
    logger.info('Initilized logger successful.')
    logger.info('Current model in %s' % config.output_dir)

    # 3. save config as own
    self.config = config

  def run(self):
    """Target"""
    # For CNN
    if self.config.target == 'cnn.classification':
      from issue.cnn.classification import classification as App
    elif self.config.target == 'cnn.regression':
      from issue.cnn.regression import regression as App
    elif self.config.target == 'cnn.pairwise':
      from issue.cnn.pairwise import pairwise as App
    # For RNN
    elif self.config.target == 'rnn.regression':
      from issue.rnn.regression import regression as App
    # For GAN
    elif self.config.target == 'gan.dcgan':
      from issue.gan.dcgan import DCGAN as App
    elif self.config.target == 'gan.cgan':
      from issue.gan.cgan import CGAN as App
    elif self.config.target == 'gan.cwgan':
      from issue.gan.cwgan import CWGAN as App
    elif self.config.target == 'gan.acgan':
      from issue.gan.acgan import ACGAN as App
    # For VAE
    elif self.config.target == 'vae.cvae':
      from issue.vae.cvae import CVAE as App
    elif self.config.target == 'vae.cvae.gan':
      from issue.vae.cvae_gan import CVAE_GAN as App
    elif self.config.target == 'vae.kinvae':
      from issue.vae.kinvae import KIN_VAE as App
    elif self.config.target == 'vae.kinvae.pair':
      from issue.vae.kinvae_pair import KIN_VAE_PAIR as App
    # For ML
    elif self.config.target == 'ml.cosine.metric':
      from issue.ml.cosine_metric import cosine_metric as App
    elif self.config.target == 'ml.active.sampler':
      from issue.ml.active_sampler import active_sampler as App
    # Unkown
    else:
      raise ValueError('Unknown target [%s]' % self.config.target)

    # init
    app = App(self.config)

    # for task
    if self.config.phase == 'train':
      app.train()
    elif self.config.phase == 'test':
      app.test()
    elif self.config.phase == 'val':
      app.val()
    else:
      raise ValueError('Unknown task [%s]' % self.config.phase)


task = Gate()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-name', type=str, dest='name', default='mnist')
  args, _ = parser.parse_known_args()

  task.initilize(args.name)
  task.run()
