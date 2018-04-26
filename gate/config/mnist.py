# -*- coding: utf-8 -*-
"""
GATE FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2018/4/25

--------------------------------------------------------

MNIST DATASET

"""

from gate.config import base
from gate.config import params


class MNIST(base.ConfigBase):

  def __init__(self, config):
    """ MNIST dataset for classification
    """
    base.ConfigBase.__init__(self, config)

    """ base """
    self.name = 'mnist'
    self.target = 'cnn.classification'
    self.data_dir = '../_datasets/mnist'
    self.output_dir = None
    self.task = 'train'

    """ log """
    self.log = params.LOG(
        print_invl=20,
        save_summary_invl=20,
        save_model_invl=500,
        test_invl=500,
        val_invl=500,
        max_iter=1000000)

    """ net """
    self.net = [params.NET()]
    self.net[0].cifarnet(num_classes=10,
                         weight_decay=0.0)

    """ phase.train """
    self.train = params.Phase('train')
    # phase.train.optimizer
    self.train.lr = [params.LR()]
    self.train.lr[0].set_vstep([0.1, 0.01, 0.001], [3000, 10000])
    self.train.opt = [params.OPT()]
    self.train.opt[0].set_sgd()

    """ data.train """
    self.train.data = params.DATA(
        batchsize=32,
        entry_path='../_datasets/mnist/train.txt',
        shuffle=True)
    self.train.data.set_queue_loader(
        loader='load_image',
        reader_thread=8,
        min_queue_num=32)
    self.set_default_data_attr(self.train.data)

    """ data.test """
    self.test = params.Phase('test')
    self.test.data = params.DATA(
        batchsize=128,
        entry_path='../_datasets/mnist/test.txt',
        shuffle=False)
    self.test.data.set_queue_loader(
        loader='load_image',
        reader_thread=1,
        min_queue_num=128)
    self.set_default_data_attr(self.test.data)

  def set_default_data_attr(self, data):
    image = params.Image()
    image.set_fixed_length_image(
        channels=3,
        frames=1,
        raw_height=28,
        raw_width=28,
        output_height=28,
        output_width=28,
        preprocessing_method='cifarnet',
        gray=False)
    data.add(image)
    data.set_entry_attr((str, int), (True, False))
    data.set_label(num_classes=10)
