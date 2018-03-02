# -*- coding: utf-8 -*-
""" Datasets
    Author: Kai JIN
    Updated: 2017-11-19
"""
from core.data.audio import data_loader_for_audio
from core.data.image import data_loader_for_image
from core.data.numeric import data_loader_for_numeric


loader_maps = {
    'load_npy': data_loader_for_numeric.load_npy,
    'load_image': data_loader_for_image.load_image,
    'load_audio': data_loader_for_audio.load_audio
}


def loads(config):
  """ config
  """
  return loader_maps[config.data.loader](config)
