# -*- coding: utf-8 -*-
""" Author: Kai JIN
    Updated: 2017-11-23
"""
from core.data import database
from core.data import data_params as params


class LFW(database.DatasetBase):

  def __init__(self, extra):
    database.DatasetBase.__init__(self, extra)

    self.name = 'lfw'
    self.target = 'lfw.vae'
    self.data_dir = '../_datasets/lfw'
    self.task = 'train'
    self.output_dir = None
    self.device = '0'

    self.log = params.Log(
        print_invl=20,
        save_summaries_invl=20,
        save_model_invl=1000,
        test_invl=1000,
        val_invl=1000,
        max_iter=999999)

    """ net """
    self.net = params.Net('kin_vae')
    self.net.set_z_dim(100)

    """ train """
    self.train = params.Phase('train')

    self.train.lr = [params.LearningRate(), params.LearningRate()]
    self.train.lr[0].set_fixed(learning_rate=0.000005)
    self.train.lr[1].set_fixed(learning_rate=0.00005)

    self.train.optimizer = [params.Optimizer(), params.Optimizer()]
    self.train.optimizer[0].set_rmsprop()
    self.train.optimizer[1].set_rmsprop()

    self.train.data = params.Data(
        batchsize=32,
        entry_path='train_1.txt',
        shuffle=True,
        total_num=5400,
        loader='load_image',
        reader_thread=32)
    self.train.data = self.set_data_attr(self.train.data)

    """ test """
    self.test = params.Phase('test')
    self.test.data = params.Data(
        batchsize=100,
        entry_path='test_1.txt',
        shuffle=False,
        total_num=600,
        loader='load_image',
        reader_thread=1)
    self.test.data = self.set_data_attr(self.test.data)

    """ val """
    self.val = params.Phase('val')
    self.val.data = params.Data(
        batchsize=100,
        entry_path='train_1.txt',
        shuffle=False,
        total_num=5400,
        loader='load_image',
        reader_thread=1)
    self.val.data = self.set_data_attr(self.val.data)

  def set_data_attr(self, data):
    default_img = params.Image(
        channels=3,
        frames=1,
        raw_height=128,
        raw_width=128,
        output_height=128,
        output_width=128,
        preprocessing_method='vae.kinship',
        gray=False)
    data.entry_path = self.data_dir + '/' + data.entry_path
    data.set_image([default_img, default_img])
    data.set_entry_attr(
        entry_dtype=(str, str, int, int),
        entry_check=(True, True, False, False))
    data.set_label(num_classes=1)
    return data
