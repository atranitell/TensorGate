# -*- coding: utf-8 -*-
""" Datasets
    Author: Kai JIN
    Updated: 2017-11-19
"""

from core.database import data_loader


loader_maps = {
    'load_image_from_text': data_loader.load_image_from_text,
    'load_pair_image_from_text': data_loader.load_pair_image_from_text,
    'load_npy_from_text': data_loader.load_npy_from_text,
    'load_triple_image_with_cond': data_loader.load_triple_image_with_cond
}


def loads(config):
  """ config
  """
  return loader_maps[config.data.loader](config)
