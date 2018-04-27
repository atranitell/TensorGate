# -*- coding: utf-8 -*-
"""
GATE FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2018/4/27

--------------------------------------------------------

Kinface Tools: Generate specific protocals files.

root = '../_datasets/kinface2'
Directory Tree:
  - fd
    - fd_001_1.jpg
    - fd_001_2.jpg
    ...
    - fd_250_1.jpg
    - fd_250_2.jpg
  - fs
    - ...
  - md
    - ...
  - ms
    - ...
  - pairs
    - fd_pairs.mat
    - fs_pairs.mat
    - md_pairs.mat
    - ms_pairs.mat

Usage:
# KinfaceTool('../_datasets/kinface2').construct_kinvae_dataset()

"""

import os
import scipy.io as scio


class KinfaceTool():

  def __init__(self, root):
    self.root = root
    self.mat_list = ['pairs/fd_pairs.mat',
                     'pairs/fs_pairs.mat',
                     'pairs/md_pairs.mat',
                     'pairs/ms_pairs.mat']
    self.kin_type_map = {'fd': 0, 'fs': 1, 'md': 2, 'ms': 3}
    self.to_path(self.mat_list)
    self.check_path(self.mat_list)

  @staticmethod
  def check_path(filepaths):
    if isinstance(filepaths, list):
      for filepath in filepaths:
        if not os.path.exists(filepath):
          raise ValueError('File [%s] does not exist.' % filepath)
    else:
      if not os.path.exists(filepath):
        raise ValueError('File [%s] does not exist.' % filepath)

  def to_path(self, files):
    if isinstance(files, list):
      for i in range(len(files)):
        files[i] = os.path.join(self.root, files[i])
    else:
      files = os.path.join(self.root, files)
    return files

  def parse_protocal_from_mat(self, mat_path):
    """ for each kind of kinship, we store as a list
      (fold, path1, path2, is_pair, kin)
      (0, fd_224_1.jpg, fd_017_2.jpg, 0, 0)
    """
    kin = mat_path.split('/')[-1].split('_')[0]
    data = scio.loadmat(mat_path)['pairs']
    res = []
    for item in data:
      fold = item[0][0][0]
      is_pair = item[1][0][0]
      pair1 = item[2][0]
      pair2 = item[3][0]
      res.append((fold, pair1, pair2, is_pair, kin))
    return res

  def _construct_kinvae_entry(self, entry):
    """ entry with:  (0, fd_224_1.jpg, fd_017_2.jpg, 0, 'fd')
      is_pair: '1' -> '1', '0' -> '-1'
    Output:
      fd/fd_224_2.jpg fd/fd_224_1.jpg fd/fd_017_2.jpg fd/fd_017_1.jpg -1 0
    """
    kin = self.kin_type_map[entry[4]]
    is_pair = '1' if int(entry[3]) == 1 else '-1'
    pair1 = entry[4] + '/' + entry[1]+' '
    pair2 = entry[4] + '/' + entry[2]+' '
    _pair1 = entry[4] + '/' + entry[1].replace('_1.jpg', '_2.jpg') + ' '
    _pair2 = entry[4] + '/' + entry[2].replace('_2.jpg', '_1.jpg') + ' '
    return _pair1+pair1+pair2+_pair2+is_pair+' '+str(kin)+'\n'

  def construct_kinvae_dataset(self):
    """ generate 5-fold: train_kinvae_*.txt, test_kinvae_*.txt
        MIXED SUBJECT VERIFICATION
    """
    data = []
    for mat in self.mat_list:
      data += self.parse_protocal_from_mat(mat)
    for fold in range(1, 6):
      f_trn = open(self.to_path('train_kinvae_%d.txt' % fold), 'w')
      f_tst = open(self.to_path('test_kinvae_%d.txt' % fold), 'w')
      for entry in data:
        if entry[0] == fold:
          f_tst.write(self._construct_kinvae_entry(entry))
        else:
          f_trn.write(self._construct_kinvae_entry(entry))
      f_trn.close()
      f_tst.close()


KinfaceTool('../_datasets/kinface1').construct_kinvae_dataset()
