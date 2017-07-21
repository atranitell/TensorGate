# -*- coding: utf-8 -*-
""" updated: 2017/3/16
"""

from abc import ABCMeta, abstractmethod


class Net(metaclass=ABCMeta):

    def __init__(self, hps, reuse=None):
        self.weight_decay = hps.weight_decay
        self.dropout = hps.dropout
        self.batch_norm_decay = hps.batch_norm_decay
        self.batch_norm_epsilon = hps.batch_norm_epsilon
        self.batch_norm_scale = hps.batch_norm_scale
        self.reuse = reuse
        self.hps = hps

    @abstractmethod
    def model(self, images, num_classes, is_training):
        pass

    @abstractmethod
    def arguments_scope(self):
        pass
