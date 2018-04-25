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


class ConfigBase():

  def __init__(self, config):
    pass

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
