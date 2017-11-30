# -*- coding: utf-8 -*-
""" MNIST FOR CLASSIFICATION
    Author: Kai JIN
    Updated: 2017-11-23
"""
from config import params


class kinvae():

  def __init__(self):

    self.name = 'kinvae'
    self.target = 'vae.kinvae'
    self.data_dir = '../_datasets/kinface2'
    self.phase = 'train'
    self.output_dir = None
    self.device = '0'

    self.log = params.Log(
        print_invl=20,
        save_summaries_invl=10,
        save_model_invl=500,
        test_invl=2000,
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
        gray=False)

    self.set_phase(self.phase)

    self.net = params.Net('cgan')
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

  def _train(self):
    """ just train phase has 'lr', 'optimizer'.
    """
    self.phase = 'train'
    self.data = params.Data(
        batchsize=32,
        entry_path="../_datasets/kinface2/train.txt",
        shuffle=True,
        total_num=800,
        loader='load_pair_image_from_text')
    self.data.add_image(self.image)
    self.data.add_image(self.image)
    self.data.label(num_classes=4)

    self.lr = [params.LearningRate(),
               params.LearningRate()]
    self.lr[0].fixed(learning_rate=0.001)
    self.lr[1].fixed(learning_rate=0.001)

    self.optimizer = [params.Optimizer(),
                      params.Optimizer()]
    self.optimizer[0].adam(beta1=0.5)
    self.optimizer[1].adam(beta1=0.5)

  def _test(self):
    self.phase = 'test'
    self.data = params.Data(
        batchsize=100,
        entry_path="../_datasets/kinface2/all.txt",
        shuffle=False,
        total_num=1000,
        loader='load_pair_image_from_text',
        reader_thread=1)
    self.data.add_image(self.image)
    self.data.add_image(self.image)
    self.data.label(num_classes=4)

  def _val(self):
    self.phase = 'val'
    self.data = params.Data(
        batchsize=100,
        entry_path="../_datasets/kinface2/test.txt",
        shuffle=False,
        total_num=100,
        loader='load_pair_image_from_text',
        reader_thread=1)
    self.data.add_image(self.image)
    self.data.add_image(self.image)
    self.data.label(num_classes=4)


class kinvae_pair():

  def __init__(self):

    self.name = 'kinvae.pair'
    self.target = 'vae.kinvae.pair'
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
    self.net.set_weight_decay(0.0001)
    self.net.set_dropout_keep(0.5)
    self.net.set_batch_norm(0.9)
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

  def _train(self):
    """ just train phase has 'lr', 'optimizer'.
    """
    self.phase = 'train'
    self.data = params.Data(
        batchsize=32,
        entry_path="../_datasets/kinface2/ms_floder_1_train_p10.txt",
        shuffle=True,
        total_num=4000,
        loader='load_triple_image_from_text')
    self.data.add_image(self.image)
    self.data.add_image(self.image)
    self.data.add_image(self.image)
    self.data.label(num_classes=1)

    self.lr = [params.LearningRate(),
               params.LearningRate(),
               params.LearningRate()]
    self.lr[0].fixed(learning_rate=0.0002)
    self.lr[1].fixed(learning_rate=0.001)
    self.lr[2].fixed(learning_rate=0.01)

    self.optimizer = [params.Optimizer(),
                      params.Optimizer(),
                      params.Optimizer()]
    self.optimizer[0].adam(beta1=0.5)
    self.optimizer[1].adam(beta1=0.5)
    self.optimizer[2].sgd()

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
    self.data.label(num_classes=1)

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
    self.data.label(num_classes=1)
