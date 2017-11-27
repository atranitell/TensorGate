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
    self.data_dir = 'E:/Datasets/TrafficNet'
    self.phase = 'train'
    self.output_dir = None
    self.device = '0'

    self.log = params.Log(
        print_invl=20,
        save_summaries_invl=20,
        save_model_invl=500,
        test_invl=500,
        val_invl=500,
        max_iter=999999)

    self.image = params.Image(
        channels=10,
        frames=1,
        raw_height=112,
        raw_width=112,
        preprocessing_method=None)

    self.set_phase(self.phase)

    self.net = params.Net()
    self.net.resnet('resnet_v2_50')

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
        entry_path="E:/Datasets/TrafficNet/data_0.6_train.txt",
        shuffle=True,
        total_num=21642,
        loader='load_npy_from_text')
    self.data.add_image(self.image)
    self.data.label(num_classes=1, span=1)

    self.lr = [params.LearningRate()]
    self.lr[0].fixed(learning_rate=0.1)

    self.optimizer = [params.Optimizer()]
    self.optimizer[0].adam(beta1=0.9)

  def _test(self):
    self.phase = 'test'
    self.data = params.Data(
        batchsize=50,
        entry_path="E:/Datasets/TrafficNet/data_0.6_test.txt",
        shuffle=False,
        total_num=5410,
        loader='load_npy_from_text')
    self.data.add_image(self.image)
    self.data.label(num_classes=1, span=1)
