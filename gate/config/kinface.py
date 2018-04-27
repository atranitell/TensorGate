# -*- coding: utf-8 -*-
"""
GATE FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2018/4/25

--------------------------------------------------------

KINFACE DATASET

"""

from gate.config import base
from gate.config import params


class KinfaceVAE(base.ConfigBase):

  def __init__(self, config):
    """ Kinface dataset for classification
    """
    base.ConfigBase.__init__(self, config)
    r = self._read_config_file

    """ base """
    self.name = r('kinface2.vae', 'name')
    self.target = r('kinface.1E', 'target')
    self.data_dir = r('../_datasets/kinface2', 'data_dir')
    self.task = r('train', 'task')
    self.output_dir = r(None, 'output_dir')
    self.device = '0'

    """ log """
    self.log = params.LOG(
        print_invl=20,
        save_summary_invl=20,
        save_model_invl=500,
        test_invl=500,
        val_invl=500,
        max_iter=15000)

    """ net """
    self.net = [params.NET()]
    self.net[0].kinvae(z_dim=100)

    """ phase.train """
    self.train = params.Phase('train')
    # phase.train.optimizer
    self.train.lr = [params.LR(), params.LR()]
    self.train.lr[0].set_fixed(learning_rate=r(0.000005, 'train.lr0'))
    self.train.lr[1].set_fixed(learning_rate=r(0.00005, 'train.lr1'))
    self.train.opt = [params.OPT(), params.OPT()]
    self.train.opt[0].set_rmsprop()
    self.train.opt[1].set_rmsprop()

    """ data.train """
    self.train.data = params.DATA(
        batchsize=r(32, 'train.batchsize'),
        entry_path=r('../_datasets/kinface2/train_1.txt', 'train.entry_path'),
        shuffle=True)
    self.train.data.set_queue_loader(
        loader='load_image',
        reader_thread=8,
        min_queue_num=32)
    self.set_default_data_attr(self.train.data)

    """ data.val """
    self.val = params.Phase('val')
    self.val.data = params.DATA(
        batchsize=r(100, 'test.batchsize'),
        entry_path=r('../_datasets/kinface2/train_1.txt', 'val.entry_path'),
        shuffle=False)
    self.val.data.set_queue_loader(
        loader='load_image',
        reader_thread=1,
        min_queue_num=128)
    self.set_default_data_attr(self.val.data)

    """ data.test """
    self.test = params.Phase('test')
    self.test.data = params.DATA(
        batchsize=r(100, 'test.batchsize'),
        entry_path=r('../_datasets/kinface2/test_1.txt', 'test.entry_path'),
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
        raw_height=64,
        raw_width=64,
        output_height=64,
        output_width=64,
        preprocessing_method='vae.kinship',
        gray=False)
    data.add([image, image, image, image])
    data.set_entry_attr((str, str, str, str, int, int),
                        (True, True, True, True, False, False))
    data.set_label(num_classes=4)
