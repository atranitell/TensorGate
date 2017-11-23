# -*- coding: utf-8 -*-
""" Datasets
    Author: Kai JIN
    Updated: 2017-11-19
"""

from core.database import data_loader


loader_maps = {
    'load_image_from_text': data_loader.load_image_from_text
}


def loads(config):
  """ config
  """
  return loader_maps[config.data.loader](config)
  # return data_loader.load_pair_image_from_text(self.config, self.phase)
