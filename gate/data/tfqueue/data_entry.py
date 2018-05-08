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
"""Data Entry"""

import os
from gate.utils import filesystem
from gate.utils.logger import logger


def parse_from_text(text_path, dtype_list, path_list):
  """ dtype_list is a tuple, which represent a list of data type.

  Example:
    The file format like:
        a/1.jpg 3 2.5
        a/2.jpg 4 3.4
    dtype_list: (str, int, float)
    path_list: (true, false, false)

  Returns:
    res: according to the dtype_list, return a tuple and each item is a list.
    count: a total number of accepted data.

  """
  logger.start_timer()
  filesystem.raise_path_not_exist(text_path)

  dtype_size = len(dtype_list)
  assert dtype_size == len(path_list)

  # show
  logger.sys('Parse items from text file %s' % text_path)

  # construct the value to return and store
  res = []
  for _ in range(dtype_size):
    res.append([])

  # start to parse
  count = 0
  with open(text_path, 'r') as fp:
    for line in fp:
      # check content number
      r = line[:-1].split(' ')
      if len(r) != dtype_size:
        continue
      # check path
      # transfer type
      for idx, dtype in enumerate(dtype_list):
        val = dtype(r[idx])
        if path_list[idx]:
          val = os.path.join(os.path.dirname(text_path), val)
          filesystem.raise_path_not_exist(val)
        res[idx].append(val)
      # count
      count += 1

  logger.end_timer('Total loading in %d files, ' % count)
  return res, count
