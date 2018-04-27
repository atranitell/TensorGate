# -*- coding: utf-8 -*-
"""
GATE FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2018/4/19

--------------------------------------------------------

CONFIG BASE

"""

import json


class ConfigBase():

  def __init__(self, config):
    self.config_value = self._load_config_file(config)

  def set_phase(self, phase):
    """ for switch phase
    """
    if phase == 'train':
      self._train()
    elif phase == 'test':
      self._test()
    elif phase == 'val':
      self._val()
    else:
      raise ValueError('Unknown phase [%s]' % phase)

  def _train(self):
    self.phase = self.train.name
    self.data = self.train.data
    self.lr = self.train.lr
    self.opt = self.train.opt

  def _test(self):
    self.phase = self.test.name
    self.data = self.test.data

  def _val(self):
    self.phase = self.val.name
    self.data = self.val.data

  def _load_config_file(self, config):
    if config is not None:
      with open(config) as fp:
        return json.load(fp)
    else:
      return None

  def _read_config_file(self, default_v, key_v):
    """ key_v like 'train.lr' -> [train][lr]
    """
    r = key_v.split('.')
    try:
      if len(r) == 1:
        config_v = self.config_value[r[0]]
      elif len(r) == 2:
        config_v = self.config_value[r[0]][r[1]]
      elif len(r) == 3:
        config_v = self.config_value[r[0]][r[1]][r[2]]
      elif len(r) == 4:
        config_v = self.config_value[r[0]][r[1]][r[2]][r[3]]
      else:
        raise ValueError('Too long to implement!')
      v = config_v
    except:
      v = default_v
    return v
