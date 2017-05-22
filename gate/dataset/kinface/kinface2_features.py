# -*- coding: utf-8 -*-
""" updated: 2017/5/22
"""
from gate.data import database
from gate.data import data_param
from gate.data import data_loader


class kinface2_feature(database.Database):

    def loads(self):
        return data_loader.load_pair_numeric_data_from_npy(
            self.data_path, self.shuffle, self.data_type,
            self.min_queue_num, self.batch_size, self.reader_thread)

    def __init__(self, data_type, name, chkp_path=None):
        # init basic class
        database.Database.__init__(self)

        self.data_type = data_type
        self.name = name
        self.num_classes = 2

        # share info
        self.min_queue_num = 64
        self._set_phase(data_type)

        # hps
        self.net_name = 'mlp'
        self.hps = data_param.hps(self.net_name)
        self.hps.set_dropout(0.5)
        self.hps.set_weight_decay(0.002)

        # log
        self.log = data_param.log(self.data_type, self.name, chkp_path)
        self.log.set_log(
            print_frequency=20,
            save_summaries_iter=20,
            save_model_iter=200,
            test_interval=200)

        # optimizer
        self.opt = data_param.optimizer()
        self.opt.set_sgd()

        # lr
        self.lr = data_param.learning_rate()
        self.lr.set_fixed(learning_rate=0.01)

        self._print()

    def _test(self):
        self.batch_size = 50
        self.total_num = 400
        self.name = self.name + '_test'
        self.reader_thread = 1
        self.shuffle = False
        self.data_path = '../_datasets/kinface2_feature/val_1_idx.txt'

    def _val(self):
        # basic param
        self.batch_size = 50
        self.total_num = 1600
        self.name = self.name + '_val'
        self.reader_thread = 1
        self.shuffle = False
        self.data_path = '../_datasets/kinface2_feature/train_1_idx.txt'

    def _train(self):
        # basic param
        self.batch_size = 50
        self.total_num = 1600
        self.name = self.name + '_train'
        self.reader_thread = 4
        self.shuffle = True
        self.data_path = '../_datasets/kinface2_feature/train_1_idx.txt'
