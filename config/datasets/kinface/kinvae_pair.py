# -*- coding: utf-8 -*-
""" Author: Kai JIN
    Updated: 2017-11-23
"""
from config import params
import core.utils.path as path


class KinfaceBase():

  def __init__(self):
    self.log = params.Log(
        print_invl=20,
        save_summaries_invl=20,
        save_model_invl=500,
        test_invl=500,
        val_invl=500,
        max_iter=10000)
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

  def _train(self):
    self.phase = 'train'
    self.data = params.Data(
        batchsize=None,
        entry_path=None,
        shuffle=True,
        total_num=None,
        loader='load_image',
        reader_thread=1)
    self.default_data_attr()
    self.default_lr()
    self.default_optimizer()

  def _test(self):
    self.phase = 'test'
    self.data = params.Data(
        batchsize=1,
        entry_path=None,
        shuffle=False,
        total_num=None,
        loader='load_image',
        reader_thread=1)
    self.default_data_attr()

  def _val(self):
    self.phase = 'val'
    self.data = params.Data(
        batchsize=1,
        entry_path=None,
        shuffle=False,
        total_num=None,
        loader='load_image',
        reader_thread=1)
    self.default_data_attr()

  def default_data_attr(self):
    self.data.set_image([self.default_image(),
                         self.default_image(),
                         self.default_image(),
                         self.default_image()])
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
    self.optimizer[0].set_rmsprop()
    self.optimizer[1].set_rmsprop()

  @staticmethod
  def parse_extra(info=None):
    """ split extra inforamtion 
    -extra=fold.target
    e.g. -extra=1.encoder # use encoder to train fold1
         -extra=2 # use bidirect to train fold2
    """
    if info is None:
      fold = '1'
      target = '.bidirect'
    else:
      res = info.split('.')
      size = len(res)
      if size == 1:
        fold = res[0]
        target = '.bidirect'
      elif size == 2:
        fold = res[0]
        target = '.' + res[1]
      else:
        raise ValueError('Unknown Parse [%s]' % info)
    return fold, target


class kinvae1_pair(KinfaceBase):

  def __init__(self, extra):
    self._fold, self._target = self.parse_extra(extra)
    self.name = 'kinvae1.pair' + self._fold
    self.target = 'kinvae' + self._target
    self.data_dir = '../_datasets/kinface1'
    self.phase = 'train'
    self.output_dir = None
    self.device = '0'
    KinfaceBase.__init__(self)

  def _train(self):
    """ just train phase has 'lr', 'optimizer'.
    """
    KinfaceBase._train(self)
    self.data.batchsize = 32
    self.data.entry_path = path.join(self.data_dir, 'train_' + self._fold)
    self.data.total_num = 854

  def _test(self):
    KinfaceBase._test(self)
    self.data.entry_path = path.join(self.data_dir, 'test_' + self._fold)
    self.data.total_num = 212

  def _val(self):
    KinfaceBase._val(self)
    self.data.entry_path = path.join(self.data_dir, 'train_' + self._fold)
    self.data.total_num = 854


class kinvae2_pair(KinfaceBase):

  def __init__(self, extra):
    self._fold, self._target = self.parse_extra(extra)
    self.name = 'kinvae2.pair' + self._fold
    self.target = 'kinvae' + self._target
    self.data_dir = '../_datasets/kinface2'
    self.phase = 'train'
    self.output_dir = None
    self.device = '0'
    KinfaceBase.__init__(self)

  def _train(self):
    """ just train phase has 'lr', 'optimizer'.
    """
    KinfaceBase._train(self)
    self.data.batchsize = 32
    self.data.total_num = 1600
    self.data.entry_path = path.join(self.data_dir, 'train_' + self._fold)

  def _test(self):
    KinfaceBase._test(self)
    self.data.batchsize=100
    self.data.total_num = 400
    self.data.entry_path = path.join(self.data_dir, 'test_' + self._fold)

  def _val(self):
    KinfaceBase._val(self)
    self.data.batchsize=100
    self.data.total_num = 1600
    self.data.entry_path = path.join(self.data_dir, 'train_' + self._fold)