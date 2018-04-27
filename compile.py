# -*- coding: utf-8 -*-
"""
GATE FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2017/12/09

--------------------------------------------------------

A tootls compile py to pyc and mv all py to another folder

"""

import os
import argparse
import shutil
import py_compile

# compile config - processing *.py files
_ROOT = '../'
_COMPILE_EXCLUDE_ROOT = ['_', '.git', '.vscode']  # skip folders
_COMPILE_EXCLUDE_FILE = ['']  # skip files


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


def process(root, fname, dir_bin=None, dir_src=None):
  """ compile and move """
  path = os.path.join(root, fname)
  if dir_bin is not None:
    dst_bin = path.replace('./', dir_bin + '/') + 'c'
    py_compile.compile(path, cfile=dst_bin, optimize=2)
    print(dst_bin)

  if dir_src is not None:
    dst_src = path.replace('./', dir_src + '/')
    if not os.path.exists(os.path.dirname(dst_src)):
      os.makedirs(os.path.dirname(dst_src))
    shutil.copy(path, dst_src)
    print(dst_src)


def traverse(dir_bin=None, dir_src=None):
  """ traverse tree """
  if dir_bin is not None:
    dir_bin = os.path.join(_ROOT, dir_bin)
    mkdir(dir_bin)
  if dir_src is not None:
    dir_src = os.path.join(_ROOT, dir_src)
    mkdir(dir_src)
  for paths in os.walk('./'):
    root = paths[0]
    if is_include(root, _COMPILE_EXCLUDE_ROOT):
      continue
    for fname in paths[2]:
      if is_include(fname, _COMPILE_EXCLUDE_FILE):
        continue
      if os.path.splitext(fname)[1] == '.py':
        process(root, fname, dir_bin, dir_src)


if __name__ == '__main__':
  # parse command line
  parser = argparse.ArgumentParser()
  parser.add_argument('-bin', type=str, dest='bin', default=None)
  parser.add_argument('-src', type=str, dest='src', default=None)
  args, _ = parser.parse_known_args()
  traverse(args.bin, args.src)
