# -*- coding: utf-8 -*-
""" updated: 2017/3/16
"""
from gate.data import database
from gate.data import data_param
from gate.data import data_loader


class cifar10(database.Database):

    def loads(self):
        return data_loader.load_image_from_text(
            self.data_path, self.shuffle, self.data_type, self.image,
            self.min_queue_num, self.batch_size, self.reader_thread)

    def __init__(self, data_type, name, chkp_path=None):
        # init basic class
        database.Database.__init__(self)

        # basic info
        self.data_type = data_type
        self.name = name
        self.num_classes = 10
        self.min_queue_num = 256
        self._set_phase(data_type)

        # image
        self.image = data_param.image()
        self.image.set_format(channels=3)
        self.image.set_raw_size(32, 32)
        self.image.set_output_size(32, 32)
        self.image.set_preprocessing('cifarnet')

        # log
        self.log = data_param.log(data_type, name, chkp_path)
        self.log.set_log(print_frequency=20,
                         save_summaries_iter=20,
                         save_model_iter=400,
                         test_interval=400)

        # setting hps
        self.hps = data_param.hps('resnet_cifar')
        self.hps.set_weight_decay(0.002)
        # self.hps.set_dropout(0.5)

        # optimizer
        self.opt = data_param.optimizer()
        self.opt.set_momentum(0.9)

        # lr
        self.lr = data_param.learning_rate()
        # self.lr.set_exponential(0.1, 350.0, 0.1)
        self.lr.set_vstep([0.01, 0.1, 0.01, 0.001], [400, 32000, 48000])

        # show
        self._print()

    def _test(self):
        self.batch_size = 100
        self.total_num = 10000
        self.name = self.name + '_test'
        self.reader_thread = 32
        self.shuffle = False
        self.data_path = '../_datasets/cifar10/test.txt'

    def _train(self):
        # basic param
        self.batch_size = 128
        self.total_num = 50000
        self.name = self.name + '_train'
        self.reader_thread = 32
        self.shuffle = True
        self.data_path = '../_datasets/cifar10/train.txt'
