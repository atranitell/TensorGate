# -*- coding: utf-8 -*-
""" trafficflow FOR CLASSIFICATION
    Author: Kai JIN
    Updated: 2017-11-23
"""
from config import params


class trafficflow():

  def __init__(self):

    self.name = 'trafficflow'
    self.target = 'cnn.regression'
    self.data_dir = '../_datasets/TrafficNet'
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

    self.image = params.Image(
        channels=10,
        frames=1,
        raw_height=112,
        raw_width=112,
        preprocessing_method=None)

    self.set_phase(self.phase)

    # self.net = params.Net('simplenet')
    # self.net = params.Net('vgg_16')
    self.net = params.Net('lightnet')
    self.net.set_weight_decay(0.0005)
    # self.net.set_batch_norm(use_batch_norm=False)
    # self.net.set_activation_fn('relu')
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
        batchsize=16,
        entry_path="../_datasets/TrafficNet/data_112_train.txt",
        shuffle=True,
        total_num=13661,
        loader='load_npy_from_text',
        reader_thread=16)
    self.data.add_image(self.image)
    self.data.label(num_classes=1, span=1)

    self.lr = [params.LearningRate()]
    self.lr[0].fixed(learning_rate=0.001)

    self.optimizer = [params.Optimizer()]
    self.optimizer[0].sgd()

  def _test(self):
    self.phase = 'test'
    self.data = params.Data(
        batchsize=50,
        entry_path="../_datasets/TrafficNet/data_112_test.txt",
        shuffle=False,
        total_num=3340,
        loader='load_npy_from_text',
        reader_thread=1)
    self.data.add_image(self.image)
    self.data.label(num_classes=1, span=1)
