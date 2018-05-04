# -*- coding: utf-8 -*-
"""
GATE FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2018/4/25

--------------------------------------------------------

COCO DATASET

"""

from gate.config import base
from gate.config import params


class COCO2014(base.ConfigBase):

  def __init__(self, config):

    base.ConfigBase.__init__(self, config)

    """ base """
    self.name = 'coco2014'
    self.target = 'detect.mask_rcnn'
    self.data_dir = '../_datasets/coco2014'
    self.output_dir = None
    self.task = 'train'

    """ log """
    self.log = params.LOG(
        print_invl=2,
        save_summary_invl=20,
        save_model_invl=1000,
        test_invl=1000,
        val_invl=1000,
        max_iter=120000)

    """ backbone network """
    self.net = [params.NET()]
    self.net[0].resnet_v2(
        depth='50',
        num_classes=1001,
        weight_decay=0.0005,
        batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5,
        batch_norm_scale=True,
        use_batch_norm=True,
        activation_fn='relu',
        global_pool=True)

    """ phase.train """
    self.train = params.Phase('train')
    # phase.train.optimizer
    self.train.lr = [params.LR()]
    self.train.lr[0].set_fixed(learning_rate=0.001)
    self.train.opt = [params.OPT()]
    self.train.opt[0].set_momentum(0.9)

    """ data.train """
    self.train.data = params.DATA(
        batchsize=1,
        entry_path='../_datasets/coco2014/instances_train2014.json',
        shuffle=True)
    self.train.data.set_custom_loader(
        loader='coco',
        data_dir='../_datasets/coco2014/train2014')

    """ data.val """
    self.val = params.Phase('val')
    self.val.data = params.DATA(
        batchsize=1,
        entry_path='../_datasets/coco2014/instances_val2014.json',
        shuffle=False)
    self.val.data.set_custom_loader(
        loader='coco',
        data_dir='../_datasets/coco2014/val2014')

    """ data.test """
    self.test = params.Phase('test')
    self.test.data = params.DATA(
        batchsize=1,
        entry_path='../_datasets/coco2014/instances_minival2014.json',
        shuffle=False)
    self.test.data.set_custom_loader(
        loader='coco',
        data_dir='../_datasets/coco2014/val2014')
