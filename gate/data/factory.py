# -*- coding: utf-8 -*-
"""
DECTRION FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2018/4/19

--------------------------------------------------------

Data Loader

"""

from gate.data.queue import data_loader


loader_maps = {
    'load_image': data_loader.load_image,
}


def get_data(config):
  """ config
  """
  return loader_maps[config.data.loader](config)
