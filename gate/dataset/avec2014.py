# -*- coding: utf-8 -*-
""" updated: 2017/3/16
"""

from gate.data import database
from gate.data import data_param
from gate.data import data_loader


class avec2014(database.Database):

    def loads(self):
        return data_loader.load_image_from_text(
            self.data_path, self.shuffle, self.data_type,
            self.frames, self.channels, self.preprocessing_method,
            self.raw_height, self.raw_width,
            self.output_height, self.output_width,
            self.min_queue_num, self.batch_size, self.reader_thread)

    def __init__(self, data_type, name):
        # basic info
        self.data_type = data_type
        self.name = name
        self.frames = 1
        self.channels = 3
        self.raw_height = 256
        self.raw_width = 256
        self.output_height = 28
        self.output_width = 28
        self.min_queue_num = 128
        self.num_classes = 63
        self.preprocessing_method = 'avec2014'

        # specify information
        # img / seq / succ
        self.avec2014_error_type = 'img'

        # hps
        self.net_name = 'cifarnet'
        self.hps = data_param.hps(self.net_name)
        self.hps.set_dropout(0.5)
        self.hps.set_weight_decay(0.0005)
        self.hps.set_batch_norm(
            batch_norm_decay=0.997,
            batch_norm_epsilon=1e-5)

        if data_type == 'train':
            self._train()
        elif data_type == 'test':
            self._test()
        else:
            raise ValueError('Unknown command %s' % self.data_type)

        # log
        self.log = data_param.log(
            self.data_type, self.name + '_' + self.net_name)
        self.log.set_log(
            print_frequency=50,
            save_summaries_iter=50,
            save_model_iter=200,
            test_interval=200)

        # show
        self._print()

    def _test(self):
        self.batch_size = 32
        # 0-5503, 1-6195, 2-5740, 3-5394, 4-6235
        # 17727
        self.total_num = 1000
        self.name = self.name + '_test'
        self.reader_thread = 1
        self.shuffle = False
        self.data_path = '../_datasets/AVEC2014/pp_tst_img.txt'

    def _train(self):
        self.batch_size = 4
        # 0-23564, 1-22872, 2-23327, 3-23673, 4-22832
        self.total_num = 23564
        self.name = self.name + '_train'
        self.reader_thread = 1
        self.shuffle = True
        self.data_path = '../_datasets/AVEC2014/pp_trn_0_img.txt'

        # optimizer
        self.opt = data_param.optimizer()
        self.opt.set_adam(
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8)

        # lr
        self.lr = data_param.learning_rate(0.999)
        self.lr.set_fixed(learning_rate=0.001)
