# -*- coding: utf-8 -*-
""" Author: Kai JIN
    Updated: 2017-11-23
"""
from config import params


class mnist():

  def __init__(self):

    self.name = 'mnist'
    self.target = 'cnn.classification'
    self.data_dir = '../_datasets/mnist'
    self.phase = 'train'
    self.output_dir = None
    self.device = '0'

    self.log = params.Log(
        print_invl=20,
        save_summaries_invl=10,
        save_model_invl=100,
        test_invl=100,
        val_invl=100,
        max_iter=999999)

    self.image = params.Image(
        channels=3,
        frames=1,
        raw_height=28,
        raw_width=28,
        output_height=28,
        output_width=28,
        preprocessing_method='cifarnet')

    self.set_phase(self.phase)

    self.net = params.Net('cifarnet')
    self.net.set_weight_decay(0.0001)
    self.net.set_dropout_keep(0.5)

  def set_phase(self, phase):
    """ for switch phase
    """
    if phase == 'train':
      self._train()
    elif phase == 'test':
      self._test()
    else:
      raise ValueError('Unknown phase [%s]' % phase)

  def _train(self):
    """ just train phase has 'lr', 'optimizer'.
    """
    self.phase = 'train'
    self.data = params.Data(
        batchsize=32,
        entry_path="../_datasets/mnist/train.txt",
        shuffle=True,
        total_num=55000,
        loader='load_image_from_text')
    self.data.add_image(self.image)
    self.data.label(num_classes=10)

    self.lr = [params.LearningRate()]
    self.lr[0].fixed(learning_rate=0.1)

    self.optimizer = [params.Optimizer()]
    self.optimizer[0].adam(beta1=0.9)

  def _test(self):
    self.phase = 'test'
    self.data = params.Data(
        batchsize=50,
        entry_path="../_datasets/mnist/test.txt",
        shuffle=True,
        total_num=10000,
        loader='load_image_from_text')
    self.data.add_image(self.image)
    self.data.label(num_classes=10)





