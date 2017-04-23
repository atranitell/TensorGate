# -*- coding: utf-8 -*-
""" updated: 2017/3/16
"""

# from gate.data import data_model
from gate.data import database
from gate.data import data_param
from gate.data import data_loader


class avec2014_fuse_16f_succ(database.Database):

    def loads(self):
        return data_loader.load_pair_block_succ_video_from_text(
            self.data_path, self.shuffle, self.data_type, self.frames, self.channels,
            self.preprocessing_method1, self.preprocessing_method2,
            self.raw_height, self.raw_width,
            self.output_height, self.output_width,
            self.min_queue_num, self.batch_size, self.reader_thread)

    def __init__(self, data_type, name):
        self.data_type = data_type
        self.name = name
        self.frames = 16
        self.channels = 3
        self.raw_height = 256
        self.raw_width = 256
        self.output_height = 224
        self.output_width = 224
        self.min_queue_num = 4
        self.device = '/gpu:0'
        self.num_classes = 63
        self.preprocessing_method = None
        self.preprocessing_method1 = 'avec2014'
        self.preprocessing_method2 = 'avec2014_flow'

        if data_type == 'train':
            self._train()
        elif data_type == 'test':
            self._test()
        else:
            raise ValueError('Unknown command %s' % self.data_type)

        # log
        self.log = data_param.log(self.data_type, self.name)
        self.log.set_log(
            print_frequency=20,
            save_summaries_iter=2,
            save_model_iter=200,
            test_interval=200)

        # show
        self._print()

    def _test(self):
        self.test_file_kind = 'succ'
        self.batch_size = 1
        self.total_num = 100
        self.name = self.name + '_test'
        self.reader_thread = 1
        self.shuffle = False
        self.data_path = '../_datasets/AVEC2014/pp_fuse_tst_succ.txt'

    def _train(self):
        # basic param
        self.batch_size = 1
        self.total_num = 199
        self.name = self.name + '_train'
        self.reader_thread = 1
        self.shuffle = True
        self.data_path = '../_datasets/AVEC2014/pp_fuse_trn_succ.txt'

        # optimizer
        self.opt = data_param.optimizer()
        self.opt.set_adam(
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8)

        # lr
        self.lr = data_param.learning_rate()
        self.lr.set_fixed(learning_rate=0.001)
