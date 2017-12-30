# -*- coding: utf-8 -*-
""" Author: Kai JIN
    Updated: 2017-11-23
"""
from config import params
import core.utils.path as path


class lfw():

  def __init__(self, extra):
    self._fold, self._target = self.parse_extra(extra)
    self.name = 'lfw.pair' + self._fold
    self.target = 'kinvae' + self._target
    self.data_dir = '../_datasets/lfw'
    self.phase = 'train'
    self.output_dir = None
    self.device = '0'

    self.log = params.Log(
        print_invl=20,
        save_summaries_invl=20,
        save_model_invl=1000,
        test_invl=1000,
        val_invl=1000,
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
        raw_height=128,
        raw_width=128,
        output_height=128,
        output_width=128,
        preprocessing_method='vae.kinship',
        gray=False)

  def _train(self):
    self.phase = 'train'
    self.data = params.Data(
        batchsize=32,
        entry_path=path.join(self.data_dir, 'train_' + self._fold),
        shuffle=True,
        total_num=5400,
        loader='load_image',
        reader_thread=32)
    self.default_data_attr()
    self.default_lr()
    self.default_optimizer()

  def _test(self):
    self.phase = 'test'
    self.data = params.Data(
        batchsize=100,
        entry_path=path.join(self.data_dir, 'test_' + self._fold),
        shuffle=False,
        total_num=600,
        loader='load_image',
        reader_thread=1)
    self.default_data_attr()

  def _val(self):
    self.phase = 'val'
    self.data = params.Data(
        batchsize=100,
        entry_path=path.join(self.data_dir, 'train_' + self._fold),
        shuffle=False,
        total_num=5400,
        loader='load_image',
        reader_thread=1)
    self.default_data_attr()

  def default_data_attr(self):
    self.data.set_image([self.default_image(),
                         self.default_image()])
    self.data.set_entry_attr(
        entry_dtype=(str, str, int, int),
        entry_check=(True, True, False, False))
    self.data.set_label(num_classes=1)

  def default_lr(self):
    self.lr = [params.LearningRate(),
               params.LearningRate()]
    self.lr[0].set_fixed(learning_rate=0.000005)
    self.lr[1].set_fixed(learning_rate=0.00005)

  def default_optimizer(self):
    self.optimizer = [params.Optimizer(),
                      params.Optimizer()]
    self.optimizer[0].set_rmsprop()
    self.optimizer[1].set_rmsprop()

  def parse_extra(self, info=None):
    """ split extra inforamtion 
    -extra=fold.target
    """
    if info is None:
      fold = '1'
      target = '.lfw'
    else:
      res = info.split('.')
      size = len(res)
      if size == 1:
        fold = res[0]
        target = '.lfw'
      elif size == 2:
        fold = res[0]
        target = '.' + res[1]
      else:
        raise ValueError('Unknown Parse [%s]' % info)
    return fold, target
