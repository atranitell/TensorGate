# -*- coding: utf-8 -*-
""" KINFACE
    Author: Kai JIN
    Updated: 2017-11-23
"""
from config import params


class kinface_npy():

  def __init__(self):

    self.name = 'kinface'
    self.target = 'ml.cosine.metric'
    self.data_dir = '../_datasets/kinface2'
    self.phase = 'test'
    self.output_dir = None
    self.device = '0'

    self.set_phase(self.phase)

  def set_phase(self, phase):
    """ for switch phase
    """
    if phase == 'train':
      self._train()
    elif phase == 'test':
      self._test()
    elif phase == 'val':
      self._val()
    else:
      raise ValueError('Unknown phase [%s]' % phase)

  def _test(self):
    self.phase = 'test'
    self.data = params.Data(
        batchsize=100,
        entry_path="../_datasets/kinface2/test_1_LBP.txt",
        shuffle=False,
        total_num=400,
        loader='load_npy',
        reader_thread=1)
    self.data.add_numpy(params.Numpy([3776]))
    self.data.set_entry_attr((str, str, int), (True, True, False))
    self.data.set_label(num_classes=1)

  def _val(self):
    self.phase = 'val'
    self.data = params.Data(
        batchsize=100,
        entry_path="../_datasets/kinface2/train_1_LBP.txt",
        shuffle=False,
        total_num=1600,
        loader='load_npy',
        reader_thread=1)
    self.data.add_numpy(params.Numpy([3776]))
    self.data.set_entry_attr((str, str, int), (True, True, False))
    self.data.set_label(num_classes=1)
