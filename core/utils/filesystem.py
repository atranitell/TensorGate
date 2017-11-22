# -*- coding: utf-8 -*-
""" updated: 2017/3/28
"""
import os
import shutil
from datetime import datetime


def mkdir(path, raise_path_exits=False):
  """
  Return: path
  """
  if not os.path.exists(path):
    os.mkdir(path)
  else:
    if raise_path_exits:
      raise ValueError('Path %s has existed.' % path)
  return path

