# -*- coding: utf-8 -*-
""" Datasets
    Author: Kai JIN
    Updated: 2017-11-19
"""

from core.database import data_loader


class Dataset():

  def __init__(self, config, phase):
    self.config = config
    self.phase = phase

  def loads(self):
    return data_loader.load_image_from_text(self.config, self.phase)
