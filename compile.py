# Copyright 2017 The KaiJIN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""""A tootls compile py to pyc and mv all py to another folder

Usage:
  python compile.py -bin=_bin -src=_src
"""

import os
import argparse
import shutil
import py_compile


class Compiler():

  def __init__(self, dir_bin=None, dir_src=None):
    self._EXCLUDE_ROOT = ['_', '.git', '.vscode', 'drawer', 'demo', 'asserts']
    self._EXCLUDE_FILE = None
    self.filelist = self._traverse()
    self.compile_src(dir_src)
    self.compile_bin(dir_bin)

  def compile_src(self, dir_src=None):
    # dir_src is None, do nothing
    if dir_src is None:
      return
    dir_src = os.path.join('../', dir_src)
    self.mkdir(dir_src)
    # copy to new dst
    for path in self.filelist:
      dst_src = path.replace('./', dir_src + '/')
      if not os.path.exists(os.path.dirname(dst_src)):
        os.makedirs(os.path.dirname(dst_src))
      shutil.copy(path, dst_src)
      print(dst_src)

  def compile_bin(self, dir_bin=None):
    if dir_bin is None:
      return
    dir_bin = os.path.join('../', dir_bin)
    self.mkdir(dir_bin)
    for path in self.filelist:
      dst_bin = path.replace('./', dir_bin + '/') + 'c'
      py_compile.compile(path, cfile=dst_bin, optimize=2)
      print(dst_bin)

  def _traverse(self):
    """ acquire current all python file paths
    """
    filepaths = []
    for paths in os.walk('./'):
      root = paths[0]
      if self.is_include(root, self._EXCLUDE_ROOT):
        continue
      for fname in paths[2]:
        if self.is_include(fname, self._EXCLUDE_FILE):
          continue
        if os.path.splitext(fname)[1] == '.py':
          filepaths.append(os.path.join(root, fname))
    return filepaths

  @staticmethod
  def mkdir(path):
    """ remove old and create new """
    if os.path.exists(path):
      shutil.rmtree(path)
    os.mkdir(path)

  @staticmethod
  def is_include(root, exclude):
    """ sub is a list """
    if exclude is None:
      return False
    for sub in exclude:
      if sub in root:
        return True


if __name__ == '__main__':
  # parse command line
  parser = argparse.ArgumentParser()
  parser.add_argument('-bin', type=str, dest='bin', default=None)
  parser.add_argument('-src', type=str, dest='src', default=None)
  args, _ = parser.parse_known_args()
  Compiler(args.bin, args.src)
