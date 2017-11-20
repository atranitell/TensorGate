# -*- coding: utf-8 -*-
""" Global task processing
    Author: Kai JIN
    Updated: 2017/11/19
"""

import os
import json
from datetime import datetime
from core.utils.logger import logger

OUTPUTS = './_outputs/'


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

  def _initilize(self, config_path):
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
        raise ValueError('Output dir is invalid.')

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
      from issue.cnn_classification import cnn_classification as App

    # initilization
    app = App(self.config)

    # allocate task
    if self.config['task'] == 'train':
      app.train()


task = Task()
