# -*- coding: utf-8 -*-
""" trafficflow FOR CLASSIFICATION
    Author: Kai JIN
    Updated: 2017-11-23
"""
from config import params


class trafficflow():

  def __init__(self, extra):

    self.name = 'trafficflow'
    self.target = 'ml.trafficflow'
    self.data_dir = '../_datasets/TrafficNet'
    self.phase = 'train'
    self.output_dir = None
    self.device = '0'

    self.log = params.Log(
        print_invl=20,
        save_summaries_invl=20,
        save_model_invl=500,
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

    self.net = params.Net('resnet_v2_101')
    self.net.set_dropout_keep(0.8)
    self.net.set_weight_decay(0.0001)
    self.net.set_batch_norm(0.997)
    self.net.set_initializer_fn('orthogonal')
    self.net.set_activation_fn('relu')
    self.net.set_cell_fn('lstm')
    self.net.set_units_and_layers([512, 784], 2)

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
        entry_path="../_datasets/TrafficNet/data_112_train.txt",
        shuffle=True,
        total_num=21930,
        loader='load_npy_from_text_with_cond',
        reader_thread=64)
    self.data.add_numpy(params.Numpy([112, 112, 10]))
    self.data.set_label(num_classes=1, span=1000, scale=True)

    self.lr = [params.LearningRate()]
    self.lr[0].set_fixed(learning_rate=0.0002)

    self.optimizer = [params.Optimizer()]
    self.optimizer[0].set_adam(0.9)

  def _test(self):
    self.phase = 'test'
    self.data = params.Data(
        batchsize=50,
        entry_path="../_datasets/TrafficNet/data_112_test.txt",
        shuffle=False,
        total_num=20791,
        loader='load_npy_from_text_with_cond',
        reader_thread=64)
    self.data.add_numpy(params.Numpy([112, 112, 10]))
    self.data.set_label(num_classes=1, span=1000, scale=True)
