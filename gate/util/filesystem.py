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


# def join_step(dst, step, fmt='txt', ext=''):
#   """ format: dst + '/' + '%8d' + ext + '.' + fmt
#   """
#   return os.path.join(dst, '%08d%s.%s' % (int(step), ext, fmt))


# def join(self, dst, ext, fmt='txt'):
#   """ format: dst + '/' + ext + '.' + fmt
#   """
#   return os.path.join(dst, '%s.%s' % (ext, fmt))


def filename(abspath):
  """ acquire filename with extension of a path
    automatically transfer to str type.
  Input: /p1/p2/f1.ext
  Return: f1.ext
  """
  if type(abspath) is not str:
    return os.path.split(str(abspath, encoding="utf-8"))[1]
  return os.path.split(abspath)[1]


def join_name(dst, src):
  """ e.g.
    dst = '/home/kj/tensorflow/'
    src = '/home/kj/gate/test.txt'
    ret: '/home/kj/tensorflow/text.txt'
  """
  return os.path.join(dst, filename(src))
