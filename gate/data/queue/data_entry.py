# -*- coding: utf-8 -*-
"""
GATE FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2017/4/5

--------------------------------------------------------

Data Entry

"""

import os
from gate.util.logger import logger
from gate.util import string


def parse_from_text(text_path, dtype_list, path_list):
  """ dtype_list is a tuple, which represent a list of data type.
      e.g. the file format like:
          a/1.jpg 3 2.5
          a/2.jpg 4 3.4
      dtype_list: (str, int, float)
      path_list: (true, false, false)

  Return:
      according to the dtype_list, return a tuple
      each item in tuple is a list.
  """
  # check path
  if not os.path.exists(text_path):
    raise ValueError('%s does not exist!' % text_path)

  dtype_size = len(dtype_list)
  assert dtype_size == len(path_list)

  # show
  logger.sys('Parse items from text file %s' % text_path)
  # logger.sys('Items data type: ' + string.type_list_to_str(dtype_list))

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
          if not os.path.exists(val):
            raise ValueError('%s does not exist!' % val)
        res[idx].append(val)

      # count
      count += 1

  logger.sys('Total loading in %d files.' % count)
  return res, count
