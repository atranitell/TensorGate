# -*- coding: utf-8 -*-
""" trafficflow FOR CLASSIFICATION
    Author: Kai JIN
    Updated: 2017-11-23
"""
from config import params
from config import base


class TrafficFlow(base.DatasetBase):

  def __init__(self, extra):
    base.DatasetBase.__init__(self, extra)

    self.name = 'trafficflow'
    self.target = 'ml.trafficflow'
    self.data_dir = '../_datasets/TrafficNet'
    self.task = 'train'
    self.output_dir = None
    self.device = '0'

    self.log = params.Log(
        print_invl=20,
        save_summaries_invl=20,
        save_model_invl=500,
        test_invl=1000,
        val_invl=1000,
        max_iter=999999)

    self.net = params.Net('resnet_v2_101')
    self.net.set_dropout_keep(0.8)
    self.net.set_weight_decay(0.0001)
    self.net.set_batch_norm(0.997)
    self.net.set_initializer_fn('orthogonal')
    self.net.set_activation_fn('relu')
    self.net.set_cell_fn('lstm')
    self.net.set_units_and_layers([512, 784], 2)

    """ train """
    self.train = params.Phase('train')

    self.train.lr = [params.LearningRate()]
    self.train.lr[0].set_fixed(learning_rate=0.0002)

    self.train.optimizer = [params.Optimizer()]
    self.train.optimizer[0].set_adam(0.9)

    self.train.data = params.Data(
        batchsize=32,
        entry_path="data_112_train.txt",
        shuffle=True,
        total_num=21930,
        loader='load_npy',
        reader_thread=64)
    self.train.data = self.set_data_attr(self.train.data)

    """ test """
    self.test = params.Phase('test')
    self.test.data = params.Data(
        batchsize=50,
        entry_path="data_112_test.txt",
        shuffle=False,
        total_num=20791,
        loader='load_npy',
        reader_thread=1)
    self.test.data = self.set_data_attr(self.test.data)

  def set_data_attr(self, data):
    data.entry_path = self.data_dir + '/' + data.entry_path
    data.set_entry_attr((str, float, float), (True, False))
    data.set_numpy([params.Numpy([112, 112, 10])])
    data.set_label(num_classes=1, span=1000, scale=True)
    return data
