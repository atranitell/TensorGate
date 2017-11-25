# -*- coding: utf-8 -*-
""" path helper
    updated: 2017/11/25
"""
import os


def join_step(dir1, step, fmt='txt'):
  """ step: global iteration
      fmt: file format
  """
  return os.path.join(dir1, '%08d.%s' % (int(step), fmt))


def filename(path):
  """ acquire filename with extension of a path
    automatically transfer to str type
  """
  if type(path) is not str:
    return os.path.split(str(path, encoding="utf-8"))[1]
  return os.path.split(path)[1]


def join_name(dst, src):
  """ e.g.
    dst = '/home/kj/tensorflow/'
    src = '/home/kj/gate/test.txt'
    ret: '/home/kj/tensorflow/text.txt'
  """
  return os.path.join(dst, filename(src))