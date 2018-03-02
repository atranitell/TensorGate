# -*- coding: utf-8 -*-
""" Author: Kai JIN
    Updated: 2017-11-23
"""
from core.data import database
from core.data import data_params as params


class MNIST(database.DatasetBase):

  def __init__(self, extra):
    database.DatasetBase.__init__(self, extra)
    r = self._read_config_file

    """ base """
    self.name = 'mnist'
    self.target = 'cnn.classification'  # 'ml.active.sampler'
    self.data_dir = '../_datasets/mnist'
    self.task = 'train'
    self.output_dir = None
    self.device = '0'

    """ log """
    self.log = params.Log(
        print_invl=20,
        save_summaries_invl=10,
        save_model_invl=500,
        test_invl=500,
        val_invl=500,
        max_iter=999999)

    """ net """
    self.net = params.Net('cifarnet')
    self.net.set_weight_decay(0.0001)
    self.net.set_dropout_keep(0.5)

    """ train """
    self.train = params.Phase('train')
    self.train.lr = [params.LearningRate()]
    self.train.lr[0].set_fixed(learning_rate=0.001)
    self.train.optimizer = [params.Optimizer()]
    self.train.optimizer[0].set_momentum()
    self.train.data = params.Data(
        batchsize=32,
        entry_path="train.txt",
        shuffle=True,
        total_num=55000,
        loader='load_image')
    self.train.data = self.set_data_attr(self.train.data)

    """ test """
    self.test = params.Phase('test')
    self.test.data = params.Data(
        batchsize=50,
        entry_path="test.txt",
        shuffle=False,
        total_num=10000,
        loader='load_image',
        reader_thread=1)
    self.test.data = self.set_data_attr(self.test.data)

  def set_data_attr(self, data):
    default_img = params.Image(
        channels=3,
        frames=1,
        raw_height=28,
        raw_width=28,
        output_height=28,
        output_width=28,
        preprocessing_method='cifarnet')
    data.entry_path = self.data_dir + '/' + data.entry_path
    data.set_image([default_img])
    data.set_entry_attr((str, int), (True, False))
    data.set_label(num_classes=10)
    return data


class MNISTRegression(MNIST):

  def __init__(self, extra):
    MNIST.__init__(self, extra)
    self.target = 'cnn.regression'

  def set_data_attr(self, data):
    default_img = params.Image(
        channels=3,
        frames=1,
        raw_height=28,
        raw_width=28,
        output_height=28,
        output_width=28,
        preprocessing_method='cifarnet')
    data.entry_path = self.data_dir + '/' + data.entry_path
    data.set_image([default_img])
    data.set_entry_attr((str, int), (True, False))
    data.set_label(num_classes=1, span=10, scale=True)
    return data


class MNISTGAN(database.DatasetBase):

  def __init__(self, extra):
    database.DatasetBase.__init__(self, extra)

    self.name = 'mnist'
    self.target = 'gan.acgan'
    self.data_dir = '../_datasets/mnist'
    self.task = 'train'
    self.output_dir = None
    self.device = '0'

    self.log = params.Log(
        print_invl=20,
        save_summaries_invl=10,
        save_model_invl=500,
        test_invl=500,
        val_invl=500,
        max_iter=999999)

    """ net """
    self.net = params.Net('cvae')
    self.net.set_z_dim(100)

    """ train """
    self.train = params.Phase('train')

    self.train.lr = [params.LearningRate(), params.LearningRate()]
    self.train.lr[0].set_fixed(learning_rate=0.001)
    self.train.lr[1].set_fixed(learning_rate=0.001)

    self.train.optimizer = [params.Optimizer(), params.Optimizer()]
    self.train.optimizer[0].set_adam(beta1=0.5)
    self.train.optimizer[1].set_adam(beta1=0.5)

    self.train.data = params.Data(
        batchsize=32,
        entry_path="train.txt",
        shuffle=True,
        total_num=55000,
        loader='load_image')
    self.train.data = self.set_data_attr(self.train.data)

    """ test """
    self.test = params.Phase('test')
    self.test.data = params.Data(batchsize=100)
    self.test.data.set_label(num_classes=10)

  def set_data_attr(self, data):
    default_img = params.Image(
        channels=3,
        frames=1,
        raw_height=28,
        raw_width=28,
        output_height=28,
        output_width=28,
        preprocessing_method='gan.mnist',
        gray=True)
    data.entry_path = self.data_dir + '/' + data.entry_path
    data.set_image([default_img])
    data.set_entry_attr((str, int), (True, False))
    data.set_label(num_classes=10)
    return data
