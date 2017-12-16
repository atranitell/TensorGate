# -*- coding: utf-8 -*-
""" Dataset operations
    Author: Kai JIN
"""
import random
import os
import matplotlib.pyplot as plt


def name(filename, app, ext='.txt'):
  basename = os.path.basename(filename).split('.')[0]
  return basename + '_' + app + ext


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
  basename = os.path.basename(filelist).split('.')[0]
  fw_1 = open(name(filelist, 'train'), 'w')
  fw_2 = open(name(filelist, 'test'), 'w')
  num = len(items)
  for i, c in enumerate(items):
    if i < num * ratio:
      fw_1.write(c)
    else:
      fw_2.write(c)
  fw_1.close()
  fw_2.close()


def clip(filelist, num, span, sorted_idx=2, shuffle=False):
  """ each invl select sampler <= num
    span: (start, end, invl)
      e.g. (0.1, 0.8, 0.5)
    num: select number samplers from each invl
  """
  items = []
  with open(filelist, 'r') as fp:
    for line in fp:
      v = float(line.split(' ')[sorted_idx - 1])
      items.append((v, line))
  if shuffle:
    random.shuffle(items)

  dist = []
  new_itmes = []
  for i in range(int((span[1] - span[0]) / span[2])):
    start = span[0] + i * span[2]
    end = span[0] + (i + 1) * span[2]
    count = 0
    for e in items:
      if e[0] > start and e[0] < end:
        new_itmes.append(e[1])
        dist.append(e[0])
        count += 1
      if count == num:
        break

  with open(name(filelist, 'clip'), 'w') as fw:
    for line in new_itmes:
      fw.write(line)

  remain_items = []
  for old_e in items:
    find = False
    for new_e in new_itmes:
      if new_e == old_e[1]:
        find = True
        break
    if find is False:
      remain_items.append(old_e[1])

  with open(name(filelist, 'clip_remain'), 'w') as fw:
    for line in remain_items:
      fw.write(line)


def distribution(filelist, idx=2):
  dist = []
  with open(filelist, 'r') as fp:
    for line in fp:
      v = float(line.split(' ')[idx - 1])
      dist.append(v)
  plt.hist(dist, bins=100)
  plt.grid()
  plt.show()