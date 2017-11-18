# -*- coding: utf-8 -*-
""" updated: 2017/3/16
"""
from gate.data import database
from gate.data import data_param
from gate.data import data_loader


class celeba(database.Database):

    def loads(self):
        return data_loader.load_image_from_text_multi_label(
            self.data_path, self.shuffle, self.data_type, self.num_classes,
            self.image, self.min_queue_num, self.batch_size, self.reader_thread)

    def __init__(self, data_type, name, chkp_path=None):
        # init basic class
        database.Database.__init__(self)

        # basic info
        self.data_type = data_type
        self.name = name
        self.num_classes = 40
        self.min_queue_num = 32
        self._set_phase(data_type)

        self.one_hot = True

        # image
        self.image = data_param.image()
        self.image.set_format(channels=3)
        self.image.set_raw_size(218, 178)
        self.image.set_output_size(128, 128)
        self.image.set_preprocessing('celeba_gan')

        # log
        self.log = data_param.log(data_type, name, chkp_path)
        self.log.set_log(print_frequency=50,
                         save_summaries_iter=50,
                         save_model_iter=1000,
                         test_interval=1000)

        # setting hps
        self.hps = data_param.hps('lenet')
        self.hps.set_weight_decay(0.002)
        self.hps.set_dropout(0.5)

        # optimizer
        self.opt = data_param.optimizer()
        # self.opt.set_momentum(0.9)
        self.opt.set_adam()

        # lr
        self.lr = data_param.learning_rate()
        # self.lr.set_fixed(learning_rate=0.01)
        self.lr.set_fixed(5e-5)

        # show
        self._print()

    def _test(self):
        self.batch_size = 64
        self.total_num = 10000
        self.name = self.name + '_test'
        self.reader_thread = 1
        self.shuffle = False
        self.data_path = '../_datasets/celeba/test.txt'

    def _train(self):
        # basic param
        self.batch_size = 128 # 64 for normal
        self.total_num = 202599
        self.name = self.name + '_train'
        self.reader_thread = 16
        self.shuffle = True
        self.data_path = '../_datasets/celeba/train.txt'
