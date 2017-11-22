# -*- coding: utf-8 -*-
""" Global task processing
    Author: Kai JIN
    Updated: 2017/11/19
"""

import os
import json
from datetime import datetime
from core.utils import filesystem
from core.utils.logger import logger

OUTPUTS = filesystem.mkdir('_outputs/')


class Task():
  """ Task just do such things:
  1) load config file into system
  2) create or pinpoint output directory
  3) initilize the logger system
  4) execuate task of specific target
  """

  def __init__(self):
    """ parse JSON file to py list
    """
    # avoid to conflict
    self.pid = self.get_datetime()
    self.logger = None
    self.config = None

  def initilize(self, config_path):
    # initilize global varibles
    with open(config_path) as fp:
      self.config = json.load(fp)
    # setting output directory
    self._set_output_dir()
    # init logger system
    self._logger()
    # save config file to output dir
    self._save_config()

  def _set_output_dir(self):
    if self.config['output_dir'] is None:
      output_name = OUTPUTS + self.config['name'] + '.' + \
          self.config['target'] + '.' + self.pid
      # create dir
      os.mkdir(output_name)
      self.config['output_dir'] = output_name
    else:
      if not os.path.exists(self.config['output_dir']):
        raise ValueError('Output dir %s is invalid.' %
                         self.config['output_dir'])
      # prepared for train-restore
      if 'train' in self.config:
        self.config['train']['restore'] = True

  def _logger(self):
    logger_name = self.config['task'] + '.' + self.pid + '.log'
    logger_path = os.path.join(self.config['output_dir'], logger_name)
    logger.set_filestream(logger_path)
    logger.set_screenstream()
    logger.info('Initilized logger successful.')
    self.logger = logger

  def _save_config(self):
    config_name = self.config['task'] + '.' + self.pid + '.json'
    config_path = os.path.join(self.config['output_dir'], config_name)
    with open(config_path, 'w') as fw:
      json.dump(self.config, fw)
    self.logger.info('Configuration has been saved in %s' % config_path)

  def get_datetime(self):
    return datetime.strftime(datetime.now(), '%y%m%d%H%M%S')

  def run(self):
    # setting application
    if self.config['target'] == 'cnn.classification':
      from issue.cnn.classification import classification as App

    if self.config['target'] == 'gan.dcgan':
      from issue.gan.dcgan import DCGAN as App
    elif self.config['target'] == 'gan.cgan':
      from issue.gan.cgan import CGAN as App
    elif self.config['target'] == 'gan.cwgan':
      from issue.gan.cwgan import CWGAN as App
    elif self.config['target'] == 'gan.acgan':
      from issue.gan.acgan import ACGAN as App
    elif self.config['target'] == 'gan.kingan':
      from issue.gan.kingan import KinGAN as App

    # initilization
    app = App(self.config)

    # allocate task
    if self.config['task'] == 'train':
      app.train()


task = Task()
