# -*- coding: utf-8 -*-
"""
GATE FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2018/4/25

--------------------------------------------------------

TRAFFICFLOW DATASET

"""

from gate.config import base
from gate.config import params


class TRAFFICFLOW(base.ConfigBase):

  def __init__(self, config):
    """ TRAFFICFLOW dataset for classification
    """
    base.ConfigBase.__init__(self, config)

    """ base """
    self.name = 'trafficflow'
    self.target = 'trafficflow.vanilla'
    self.data_dir = '../_datasets/trafficflow'
    self.output_dir = 'E:/gate/_outputs/trafficflow.trafficflow.vanilla.180425171913'
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
    self.net[0].resnet_v2_50(
        num_classes=1,
        weight_decay=0.0001,
        batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5,
        batch_norm_scale=True,
        use_batch_norm=True)

    """ phase.train """
    self.train = params.Phase('train')
    # phase.train.optimizer
    self.train.lr = [params.LR()]
    self.train.lr[0].set_fixed(0.001)
    self.train.opt = [params.OPT()]
    self.train.opt[0].set_adam()

    """ data.train """
    self.train.data = params.DATA(
        batchsize=32,
        entry_path='../_datasets/trafficflow/train.txt',
        shuffle=True)
    self.train.data.set_queue_loader(
        loader='load_npy',
        reader_thread=32,
        min_queue_num=32)
    self.set_default_data_attr(self.train.data)

    """ data.test """
    self.test = params.Phase('test')
    self.test.data = params.DATA(
        batchsize=100,
        entry_path='../_datasets/trafficflow/test.txt',
        shuffle=False)
    self.test.data.set_queue_loader(
        loader='load_npy',
        reader_thread=1,
        min_queue_num=128)
    self.set_default_data_attr(self.test.data)

  def set_default_data_attr(self, data):
    data.add(params.Numpy([112, 112, 10]))
    data.set_entry_attr((str, float), (True, False))
    data.set_label(num_classes=1)
