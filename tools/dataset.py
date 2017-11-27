# -*- coding: utf-8 -*-
""" Dataset operations
    Author: Kai JIN
"""
import random
import os


def divide_filelist(filelist, ratio, shuffle):
  """ partition dataset describe file
    e.g. filepath1 0
         filepath2 0
         ...
      into two part: train/test with ratio
  """
  items = []
  with open(filelist, 'r') as fp:
    for line in fp:
      items.append(line)
  if shuffle:
    random.shuffle(items)
  basename = os.path.split(filelist)[1]
  fw_1 = open(basename + '_train.txt', 'w')
  fw_2 = open(basename + '_test.txt', 'w')
  num = len(items)
  for i, c in enumerate(items):
    if i < num * ratio:
      fw_1.write(c)
    else:
      fw_2.write(c)
  fw_1.close()
  fw_2.close()
