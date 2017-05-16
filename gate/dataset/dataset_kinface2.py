# -*- coding: utf-8 -*-
""" updated: 2017/3/16
"""

from gate.data import database
from gate.data import data_param
from gate.data import data_loader


class kinface2(database.Database):

    def loads(self):
        return data_loader.load_pair_image_from_text(
            self.data_path, self.shuffle, self.data_type, self.frames, self.channels,
            self.preprocessing_method1, self.preprocessing_method2,
            self.raw_height, self.raw_width,
            self.output_height, self.output_width,
            self.min_queue_num, self.batch_size, self.reader_thread)

    def __init__(self, data_type, name):
        self.data_type = data_type
        self.name = name
        self.channels = 3
        self.frames = 1
        self.raw_height = 64
        self.raw_width = 64
        self.output_height = 28
        self.output_width = 28
        self.min_queue_num = 256
        self.device = '/gpu:0'
        self.num_classes = 2
        self.preprocessing_method = None
        self.preprocessing_method1 = 'kinface'
        self.preprocessing_method2 = 'kinface'

        if data_type == 'train':
            self._train()
        elif data_type == 'test':
            self._test()
        elif data_type == 'val':
            self._val()
        else:
            raise ValueError('Unknown command %s' % self.data_type)

        # log
        self.log = data_param.log(self.data_type, self.name)
        self.log.set_log(
            print_frequency=20,
            save_summaries_iter=20,
            save_model_iter=200,
            test_interval=200)

        # show
        self._print()

    def _test(self):
        self.batch_size = 1
        self.total_num = 100
        self.name = self.name + '_test'
        self.reader_thread = 1
        self.shuffle = False
        self.data_path = '../_datasets/kinface2/fd_train_2.txt'

    def _val(self):
        # basic param
        self.batch_size = 1
        self.total_num = 400
        self.name = self.name + '_train'
        self.reader_thread = 1
        self.shuffle = False
        self.data_path = '../_datasets/kinface2/fd_train_2.txt'

    def _train(self):
        # basic param
        self.batch_size = 32
        self.total_num = 400
        self.name = self.name + '_train'
        self.reader_thread = 4
        self.shuffle = True
        self.data_path = '../_datasets/kinface2/fd_train_2.txt'

        # optimizer
        self.opt = data_param.optimizer()
        # self.opt.set_adam(
        #     adam_beta1=0.9,
        #     adam_beta2=0.999,
        #     adam_epsilon=1e-8)
        self.opt.set_sgd()

        # lr
        self.lr = data_param.learning_rate()
        self.lr.set_exponential(
            learning_rate=0.1,
            num_epochs_per_decay=100,
            learning_rate_decay_factor=0.9
        )
        # self.lr.set_fixed(learning_rate=0.01)
