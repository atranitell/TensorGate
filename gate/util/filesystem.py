# -*- coding: utf-8 -*-
"""
DECTRION FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2018/4/21

--------------------------------------------------------

Filesystem

"""

import os
import shutil
from datetime import datetime


def mkdir(path, raise_path_exits=False):
  """
  Return: path if mkdir or path has existed
  """
  if not os.path.exists(path):
    os.mkdir(path)
  else:
    if raise_path_exits:
      raise ValueError('Path %s has existed.' % path)
  return path


def mkdirs(path, raise_path_exits=False):
  """ Create a dir leaf
  """
  if not os.path.exists(path):
    os.makedirs(path)
  else:
    if raise_path_exits:
      raise ValueError('Path %s has exitsted.' % path)
  return path


def join(*args):
  """ join multiple path
  e.g.
    join('c:', 'pp', 'c.txt')

  return:
    'c:\pp\c.txt'
  """
  assert len(args) >= 2
  ret_arg = args[0]
  for arg in args[1:]:
    ret_arg = os.path.join(ret_arg, arg)
  return ret_arg