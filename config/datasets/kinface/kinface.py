# -*- coding: utf-8 -*-
""" KINFACE
    Author: Kai JIN
    Updated: 2017-11-23
"""
from config import params


class kinface():

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
        save_model_invl=500,
        test_invl=500,
        val_invl=500,
        max_iter=999999)

    self.image = params.Image(
        channels=3,
        frames=1,
        raw_height=64,
        raw_width=64,
        output_height=64,
        output_width=64,
        preprocessing_method='vae.kinship',
        gray=True)

    self.set_phase(self.phase)

    self.net = params.Net('lightnet64')
    self.net.set_weight_decay(0.0005)
    self.net.set_dropout_keep(1.0)
    self.net.set_batch_norm(0.997)

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
        entry_path="../_datasets/kinface2/ms_floder_1_train_p101.txt",
        shuffle=True,
        total_num=4000,
        loader='load_pair_image_from_text')
    self.data.add_image(self.image)
    self.data.add_image(self.image)
    self.data.add_image(self.image)
    self.data.set_label(num_classes=1)

    self.lr = [params.LearningRate()]
    self.lr[0].set_fixed(learning_rate=0.01)

    self.optimizer = [params.Optimizer()]
    self.optimizer[0].set_sgd()

  def _test(self):
    self.phase = 'test'
    self.data = params.Data(
        batchsize=1,
        entry_path="../_datasets/kinface2/ms_floder_1_test_label.txt",
        shuffle=False,
        total_num=100,
        loader='load_pair_image_from_text',
        reader_thread=1)
    self.data.add_image(self.image)
    self.data.add_image(self.image)
    self.data.set_label(num_classes=1)

  def _val(self):
    self.phase = 'val'
    self.data = params.Data(
        batchsize=100,
        entry_path="../_datasets/kinface2/ms_floder_1_test_label.txt",
        shuffle=False,
        total_num=100,
        loader='load_pair_image_from_text',
        reader_thread=1)
    self.data.add_image(self.image)
    self.data.add_image(self.image)
    self.data.set_label(num_classes=1)
