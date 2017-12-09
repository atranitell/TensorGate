# -*- coding: utf-8 -*-
""" A tootls compile py to pyc and mv all py to another folder
  Updater: 17-12-09
  Author: Kai JIN
"""
import os
import shutil
import py_compile

DIR_BIN = '../_bin'
DIR_SRC = '../_src'

EXCLUDE_ROOT = ['_', '.git', '.vscode']
EXCLUDE_FILE = ['compile.py']


def is_include(root, exclude):
  """ sub is a list """
  for sub in exclude:
    if sub in root:
      return True


def mkdir(path):
  """ remove old and create new """
  if os.path.exists(path):
    shutil.rmtree(path)
  os.mkdir(path)


def process(root, fname):
  """ compile and move """
  path = os.path.join(root, fname)
  dst_bin = path.replace('./', DIR_BIN + '/') + 'c'
  dst_src = path.replace('./', DIR_SRC + '/')
  if not os.path.exists(os.path.dirname(dst_src)):
    os.makedirs(os.path.dirname(dst_src))
  shutil.copy(path, dst_src)
  py_compile.compile(path, cfile=dst_bin, optimize=2)


def traverse():
  """ traverse tree """
  mkdir(DIR_BIN)
  mkdir(DIR_SRC)
  for paths in os.walk('./'):
    root = paths[0]
    if is_include(root, EXCLUDE_ROOT):
      continue
    print(root)
    for fname in paths[2]:
      if is_include(fname, EXCLUDE_FILE):
        continue
      if os.path.splitext(fname)[1] == '.py':
        process(root, fname)


if __name__ == '__main__':
  traverse()
