# -*- coding: utf-8 -*-
""" KINFACE
    Author: Kai JIN
    Updated: 2017-11-23
"""
from config import params


class kinface_npy():

  def __init__(self):

    self.name = 'kinface'
    self.target = 'cnn.pairwise'
    self.data_dir = '../_datasets/kinface2'
    self.phase = 'train'
    self.output_dir = None
    self.device = '0'

    self.log = params.Log(
        print_invl=20,
        save_summaries_invl=10,
        save_model_invl=200,
        test_invl=200,
        val_invl=200,
        max_iter=999999)

    self.set_phase(self.phase)

    self.net = params.Net('mlp')

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

  def _train(self):
    """ just train phase has 'lr', 'optimizer'.
    """
    self.phase = 'train'
    self.data = params.Data(
        batchsize=32,
        entry_path="../_datasets/kinface2/train_1_facenet_webface.txt",
        shuffle=True,
        total_num=1600,
        loader='load_pair_npy_from_text')
    self.data.set_label(num_classes=1)
    self.data.add_numpy(params.Numpy([1792]))

    self.lr = [params.LearningRate()]
    self.lr[0].set_fixed(learning_rate=0.01)

    self.optimizer = [params.Optimizer()]
    self.optimizer[0].set_sgd()

  def _test(self):
    self.phase = 'test'
    self.data = params.Data(
        batchsize=100,
        entry_path="../_datasets/kinface2/test_1_facenet_webface.txt",
        shuffle=False,
        total_num=400,
        loader='load_pair_npy_from_text',
        reader_thread=1)
    self.data.add_numpy(params.Numpy([1792]))
    self.data.set_label(num_classes=1)

  def _val(self):
    self.phase = 'val'
    self.data = params.Data(
        batchsize=100,
        entry_path="../_datasets/kinface2/train_1_facenet_webface.txt",
        shuffle=False,
        total_num=1600,
        loader='load_pair_npy_from_text',
        reader_thread=1)
    self.data.add_numpy(params.Numpy([1792]))
    self.data.set_label(num_classes=1)
