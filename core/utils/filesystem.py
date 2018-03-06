# -*- coding: utf-8 -*-
""" updated: 2017/3/28
"""
import os
import shutil
from datetime import datetime


class Filesystem():
  """ filesystem related operation
  """

  def mkdir(self, path, raise_path_exits=False):
    """
    Return: path if mkdir or path has existed
    """
    if not os.path.exists(path):
      os.mkdir(path)
    else:
      if raise_path_exits:
        raise ValueError('Path %s has existed.' % path)
    return path

  def mkdirs(self, path, raise_path_exits=False):
    """ Create a dir leaf
    """
    if not os.path.exists(path):
      os.makedirs(path)
    else:
      if raise_path_exits:
        raise ValueError('Paht %s has exitsted.' % path)
    return path


filesystem = Filesystem()
