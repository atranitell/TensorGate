# -*- coding: utf-8 -*-
""" updated: 2017/3/16
"""
from gate.data import database
from gate.data import data_param
from gate.data import data_loader


class kinface2(database.Database):

    def loads(self):
        # return data_loader.load_pair_numeric_image_data(
        #     self.data_path, self.shuffle, self.data_type, self.image,
        #     self.min_queue_num, self.batch_size, self.reader_thread)
        return data_loader.load_pair_image_from_text(
            self.data_path, self.shuffle, self.data_type, self.image,
            self.min_queue_num, self.batch_size, self.reader_thread)
        # return data_loader.load_pair_image_from_text_with_multiview(
        #     self.data_path, self.shuffle, self.data_type, self.image,
        #     self.min_queue_num, self.batch_size, self.reader_thread)

    def __init__(self, data_type, name, chkp_path=None):
        # init basic class
        database.Database.__init__(self)

        # basic info
        self.data_type = data_type
        self.name = name
        self.num_classes = 2
        self.min_queue_num = 16
        self._set_phase(data_type)

        # image
        self.image = data_param.image()
        self.image.set_format(channels=3)
        self.image.set_raw_size(64, 64)
        self.image.set_output_size(32, 32)
        self.image.set_preprocessing('kinface', 'kinface')

        # log
        self.log = data_param.log(data_type, name, chkp_path)
        self.log.set_log(print_frequency=10,
                         save_summaries_iter=10,
                         save_model_iter=100,
                         test_interval=100)

        # setting hps
        self.hps = data_param.hps(net_name='lightnet32')  # mlp
        self.hps.set_weight_decay(0.002)
        self.hps.set_dropout(0.8)

        # batch norm
        self.hps.set_batch_norm(
            batch_norm_decay=0.997,
            batch_norm_epsilon=1e-5)

        # optimizer
        self.opt = data_param.optimizer()
        self.opt.set_sgd()

        # lr
        self.lr = data_param.learning_rate()
        self.lr.set_fixed(learning_rate=0.01)
        # self.lr.set_exponential(.5, 200, 0.5)

        self._print()

    def _test(self):
        self.batch_size = 50
        self.total_num = 400
        self.name = self.name + '_test'
        self.reader_thread = 1
        self.shuffle = False
        self.data_path = '../_datasets/kinface2/val_1.txt'
        # self.data_path = '../_datasets/kinface2_feature/val_5_idx.txt'

    def _val(self):
        # basic param
        self.batch_size = 50
        self.total_num = 1600
        self.name = self.name + '_val'
        self.reader_thread = 1
        self.shuffle = False
        self.data_path = '../_datasets/kinface2/train_1.txt'
        # self.data_path = '../_datasets/kinface2_feature/train_5_idx.txt'

    def _train(self):
        # basic param
        self.batch_size = 32
        self.total_num = 1600
        self.name = self.name + '_train'
        self.reader_thread = 4
        self.shuffle = True
        self.data_path = '../_datasets/kinface2/train_1.txt'
        # self.data_path = '../_datasets/kinface2_feature/train_5_idx.txt'
