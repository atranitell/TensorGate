# -*- coding: utf-8 -*-
""" Author: Kai JIN
    Updated: 2017-11-23
"""
from config import params


class kinface_base():

  def __init__(self):
    self.log = params.Log(
        print_invl=20,
        save_summaries_invl=10,
        save_model_invl=500,
        test_invl=500,
        val_invl=500,
        max_iter=999999)

    self.set_phase(self.phase)
    self.net = params.Net('kin_vae')
    self.net.set_z_dim(100)

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

  @staticmethod
  def default_image():
    return params.Image(
        channels=3,
        frames=1,
        raw_height=64,
        raw_width=64,
        output_height=64,
        output_width=64,
        preprocessing_method='vae.kinship',
        gray=False)

  def default_data_attr(self):
    self.data.add_image(self.default_image())
    self.data.add_image(self.default_image())
    self.data.add_image(self.default_image())
    self.data.add_image(self.default_image())
    self.data.set_entry_attr(
        entry_dtype=(str, str, str, str, int, int),
        entry_check=(True, True, True, True, False, False))
    self.data.set_label(num_classes=4)

  def default_lr(self):
    self.lr = [params.LearningRate(),
               params.LearningRate()]
    self.lr[0].set_fixed(learning_rate=0.00001)
    self.lr[1].set_fixed(learning_rate=0.00005)

  def default_optimizer(self):
    self.optimizer = [params.Optimizer(),
                      params.Optimizer()]
    self.optimizer[0].set_adam(0.5)
    self.optimizer[1].set_adam(0.5)


class kinvae1_pair(kinface_base):

  def __init__(self):

    self.name = 'kinvae1.pair'
    self.target = 'vae.kinvae.pair'
    self.data_dir = '../_datasets/kinface1'
    self.phase = 'train'
    self.output_dir = None
    self.device = '0'
    kinface_base.__init__(self)

  def _train(self):
    """ just train phase has 'lr', 'optimizer'.
    """
    self.phase = 'train'
    self.data = params.Data(
        batchsize=16,
        entry_path="../_datasets/kinface1/train_1.txt",
        shuffle=True,
        total_num=854,
        loader='load_image',
        reader_thread=32)
    self.default_data_attr()
    self.default_lr()
    self.default_optimizer()

  def _test(self):
    self.phase = 'test'
    self.data = params.Data(
        batchsize=1,
        entry_path="../_datasets/kinface1/test_1.txt",
        shuffle=False,
        total_num=212,
        loader='load_image',
        reader_thread=1)
    self.default_data_attr()

  def _val(self):
    self.phase = 'val'
    self.data = params.Data(
        batchsize=1,
        entry_path="../_datasets/kinface1/train_1.txt",
        shuffle=False,
        total_num=854,
        loader='load_image',
        reader_thread=1)
    self.default_data_attr()


class kinvae2_pair(kinface_base):

  def __init__(self):

    self.name = 'kinvae2.pair'
    self.target = 'vae.kinvae.bidirect'
    self.data_dir = '../_datasets/kinface2'
    self.phase = 'train'
    self.output_dir = None
    self.device = '0'
    kinface_base.__init__(self)

  def _train(self):
    """ just train phase has 'lr', 'optimizer'.
    """
    self.phase = 'train'
    self.data = params.Data(
        batchsize=32,
        entry_path="../_datasets/kinface2/train_1.txt",
        shuffle=True,
        total_num=1600,
        loader='load_image',
        reader_thread=32)
    self.default_data_attr()
    self.default_lr()
    self.default_optimizer()

  def _test(self):
    self.phase = 'test'
    self.data = params.Data(
        batchsize=100,
        entry_path="../_datasets/kinface2/test_1.txt",
        shuffle=False,
        total_num=400,
        loader='load_image',
        reader_thread=1)
    self.default_data_attr()

  def _val(self):
    self.phase = 'val'
    self.data = params.Data(
        batchsize=100,
        entry_path="../_datasets/kinface2/train_1.txt",
        shuffle=False,
        total_num=1600,
        loader='load_image',
        reader_thread=1)
    self.default_data_attr()
