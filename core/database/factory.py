# -*- coding: utf-8 -*-
""" Datasets
    Author: Kai JIN
    Updated: 2017-11-19
"""
from core.database import data_loader


loader_maps = {
    'load_npy': data_loader.load_npy,
    'load_image': data_loader.load_image
}


def loads(config):
  """ config
  """
  return loader_maps[config.data.loader](config)
