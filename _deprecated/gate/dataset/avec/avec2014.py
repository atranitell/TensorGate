# -*- coding: utf-8 -*-
""" updated: 2017/3/16
"""
from gate.data import database
from gate.data import data_param
from gate.data import data_loader


class avec2014(database.Database):

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
        self.num_classes = 63
        self.min_queue_num = 128
        self._set_phase(data_type)

        # image
        self.image = data_param.image()
        self.image.set_format(channels=3)
        # 3 view_size is (156, 156)
        self.image.set_raw_size(256, 256)
        self.image.set_output_size(224, 224)
        # 3 view is 'avec2014_3view'
        self.image.set_preprocessing('avec2014')

        # specify information
        # img / seq / succ
        # self.avec2014_error_type = 'img'

        # log
        self.log = data_param.log(data_type, name, chkp_path)
        self.log.set_log(
            print_frequency=50,
            save_summaries_iter=50,
            save_model_iter=1000,
            test_interval=1000)

        # setting hps
        self.hps = data_param.hps('resnet_v2_50')
        self.hps.set_dropout(1.0)
        self.hps.set_weight_decay(0.0005)
        self.hps.set_batch_norm(
            batch_norm_decay=0.997,
            batch_norm_epsilon=1e-5)

        # optimizer
        self.opt = data_param.optimizer()
        self.opt.set_adam(
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8)

        # lr
        self.lr = data_param.learning_rate()
        self.lr.set_fixed(learning_rate=0.001)

        self._print()

    def _test(self):
        self.batch_size = 50
        # 0-5503, 1-6195, 2-5740, 3-5394, 4-6235
        # 17727
        self.total_num = 17727
        self.name = self.name + '_test'
        self.reader_thread = 4
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
