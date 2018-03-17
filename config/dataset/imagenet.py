# -*- coding: utf-8 -*-
""" Author: Kai JIN
    Updated: 2018-03-06
"""
from core.data import database
from core.data import data_params as params


class ImageNet(database.DatasetBase):
  """ Based on images extracted from the video frames.
  """

  def __init__(self, extra):
    database.DatasetBase.__init__(self, extra)
    r = self._read_config_file

    """ base """
    self.name = 'imagenet'
    self.target = 'lab.guided-learning'  # 'cnn.classification'
    self.data_dir = '../_datasets/ImageNet'
    self.task = 'train'
    self.output_dir = '../_model/imagenet_vgg_16'
    self.device = '0'

    """ log """
    self.log = params.Log(
        print_invl=20,
        save_summaries_invl=20,
        save_model_invl=1000,
        test_invl=1000,
        val_invl=1000,
        max_iter=9999999)

    """ net """
    self.net = params.Net('vgg_16')
    self.net.set_weight_decay(0.0001)
    self.net.set_dropout_keep(0.5)
    self.net.set_batch_norm(
        batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5)

    """ train """
    self.train = params.Phase('train')
    self.train.lr = [params.LearningRate()]
    self.train.lr[0].set_fixed(0.001)
    self.train.optimizer = [params.Optimizer()]
    self.train.optimizer[0].set_sgd()

    self.train.data = params.Data(
        batchsize=1,
        entry_path="test.txt",
        shuffle=True,
        total_num=1,  # 1280000,
        loader='load_image',
        reader_thread=1)
    self.train.data = self.set_data_attr(self.train.data)

    """ test """
    self.test = params.Phase('test')
    self.test.data = params.Data(
        batchsize=1,
        entry_path="test.txt",
        shuffle=False,
        total_num=1,
        loader='load_image',
        reader_thread=1)
    self.test.data = self.set_data_attr(self.test.data)

  def set_data_attr(self, data):
    default_img = params.Image(
        channels=3,
        frames=1,
        raw_height=-1,
        raw_width=-1,
        output_height=224,
        output_width=224,
        preprocessing_method='vgg',
        gray=False)
    data.entry_path = self.data_dir + '/' + data.entry_path
    data.set_image([default_img])
    data.set_entry_attr(
        entry_dtype=(str, int),
        entry_check=(True, False))
    data.set_label(num_classes=1000)
    return data
