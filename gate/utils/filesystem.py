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
"""filesystem"""

import os


def raise_path_not_exist(path):
  """r.t."""
  if not os.path.exists(path):
    raise ValueError('Path %s not existed' % path)


def mkdir(path, raise_path_exits=False):
  """Path if mkdir or path has existed"""
  if not os.path.exists(path):
    os.mkdir(path)
  else:
    if raise_path_exits:
      raise ValueError('Path %s has existed.' % path)
  return path


def mkdirs(path, raise_path_exits=False):
  """Create a dir leaf"""
  if not os.path.exists(path):
    os.makedirs(path)
  else:
    if raise_path_exits:
      raise ValueError('Path %s has exitsted.' % path)
  return path


def join(*args):
  """Join multiple path - join('c:', 'pp', 'c.txt') -> 'c:\pp\c.txt'"""
  assert len(args) >= 2
  ret_arg = args[0]
  for arg in args[1:]:
    ret_arg = os.path.join(ret_arg, arg)
  return ret_arg


def filename(abspath):
  """Input: /p1/p2/f1.ext -> Return: f1.ext"""
  if type(abspath) is not str:
    return os.path.split(str(abspath, encoding="utf-8"))[1]
  return os.path.split(abspath)[1]


def join_name(dst, src):
  """Example: dst='/kj/tensorflow/', src='/kj/gate/test.txt'
    ret: '/kj/tensorflow/text.txt'
  """
  return os.path.join(dst, filename(src))
