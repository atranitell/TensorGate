# -*- coding: utf-8 -*-
""" Author: Kai JIN
    Updated: 2017-11-23
"""
from config import params
from config import base


class KinfaceVAE(base.DatasetBase):
  """ default for Kinface2
  """

  def __init__(self, extra):
    base.DatasetBase.__init__(self, extra)
    r = self._read_config_file

    """ base """
    self.name = r('kinface2.vae', 'name')
    self.target = r('kinvae.bidirect11', 'target')
    self.data_dir = r('../_datasets/kinface2', 'data_dir')
    self.task = r('train', 'task')
    self.output_dir = r(None, 'output_dir')
    self.device = '0'

    self.log = params.Log(
        print_invl=20,
        save_summaries_invl=20,
        save_model_invl=500,
        test_invl=500,
        val_invl=500,
        max_iter=15000)

    """ Net """
    self.net = params.Net('kin_vae')
    self.net.set_z_dim(100)

    """ train """
    self.train = params.Phase('train')

    self.train.lr = [params.LearningRate(), params.LearningRate()]
    self.train.lr[0].set_fixed(learning_rate=r(0.000005, 'train.lr0'))
    self.train.lr[1].set_fixed(learning_rate=r(0.00005, 'train.lr1'))

    self.train.optimizer = [params.Optimizer(), params.Optimizer()]
    self.train.optimizer[0].set_rmsprop()
    self.train.optimizer[1].set_rmsprop()

    self.train.data = params.Data(
        batchsize=r(32, 'train.batchsize'),
        entry_path=r('train_1.txt', 'train.entry_path'),
        shuffle=True,
        total_num=r(1600, 'train.total_num'),
        loader='load_image',
        reader_thread=32)
    self.train.data = self.set_data_attr(self.train.data)

    """ test """
    self.test = params.Phase('test')
    self.test.data = params.Data(
        batchsize=r(100, 'test.batchsize'),
        entry_path=r('test_1.txt', 'test.entry_path'),
        shuffle=False,
        total_num=r(400, 'test.total_num'),
        loader='load_image',
        reader_thread=1)
    self.test.data = self.set_data_attr(self.test.data)

    """ val """
    self.val = params.Phase('val')
    self.val.data = params.Data(
        batchsize=r(100, 'val.batchsize'),
        total_num=r(1600, 'val.total_num'),
        shuffle=False,
        entry_path=r('train_1.txt', 'val.entry_path'),
        loader='load_image',
        reader_thread=1)
    self.val.data = self.set_data_attr(self.val.data)

  def set_data_attr(self, data):
    default_img = params.Image(
        channels=3,
        frames=1,
        raw_height=64,
        raw_width=64,
        output_height=64,
        output_width=64,
        preprocessing_method='vae.kinship',
        gray=False)
    data.entry_path = self.data_dir + '/' + data.entry_path
    data.set_image([default_img] * 4)
    data.set_entry_attr(
        entry_dtype=(str, str, str, str, int, int),
        entry_check=(True, True, True, True, False, False))
    data.set_label(num_classes=4)
    return data


class KinfaceNPY(base.DatasetBase):
  """ default for Kinface2
  """

  def __init__(self, extra):
    base.DatasetBase.__init__(self, extra)

    """ base """
    self.name = 'kinface2.npy'
    self.target = 'kinvae.feature'
    self.data_dir = '../_datasets/kinface2/protocal_latest'
    self.task = 'train'
    self.output_dir = None
    self.device = '0'
    self.log = params.Log(
        print_invl=20,
        save_summaries_invl=20,
        save_model_invl=100,
        test_invl=100,
        val_invl=100,
        max_iter=15000)

    """ train """
    self.train = params.Phase('train')
    self.train.lr = [params.LearningRate()]
    self.train.lr[0].set_fixed(learning_rate=0.000005)
    self.train.optimizer = [params.Optimizer()]
    self.train.optimizer[0].set_rmsprop()
    self.train.data = params.Data(
        batchsize=32,
        entry_path="train_1_LBP.txt",
        shuffle=True,
        total_num=1600,
        loader='load_npy',
        reader_thread=32)
    self.train.data = self.set_data_attr(self.train.data)

    """ test """
    self.test = params.Phase('test')
    self.test.data = params.Data(
        batchsize=100,
        entry_path="test_1_LBP.txt",
        shuffle=False,
        total_num=400,
        loader='load_npy',
        reader_thread=1)
    self.test.data = self.set_data_attr(self.test.data)

    """ val """
    self.val = params.Phase('val')
    self.val.data = params.Data(
        batchsize=100,
        entry_path="train_1_LBP.txt",
        shuffle=False,
        total_num=1600,
        loader='load_npy',
        reader_thread=1)
    self.val.data = self.set_data_attr(self.val.data)

  def set_data_attr(self, data):
    data.entry_path = self.data_dir + '/' + data.entry_path
    data.add_numpy(params.Numpy([3776]))
    data.set_entry_attr((str, str, int), (True, True, False))
    data.set_label(num_classes=1)
    return data
