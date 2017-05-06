# -*- coding: utf-8 -*-
""" updated: 2017/5/05
"""

from gate.data import database
from gate.data import data_param
from gate.data import data_loader


class cifar10_gan(database.Database):

    def loads(self):
        return data_loader.load_image_from_text(
            self.data_path, self.shuffle, self.data_type,
            self.frames, self.channels, self.preprocessing_method,
            self.raw_height, self.raw_width,
            self.output_height, self.output_width,
            self.min_queue_num, self.batch_size, self.reader_thread)

    def __init__(self, data_type, name):
        self.data_type = data_type
        self.name = name
        self.channels = 3
        self.frames = 1
        self.raw_height = 32
        self.raw_width = 32
        self.output_height = 28
        self.output_width = 28
        self.min_queue_num = 256
        self.device = '/gpu:0'
        self.num_classes = 10
        self.preprocessing_method = 'mnist_gan'

        if data_type == 'train':
            self._train()
        elif data_type == 'test':
            self._test()
        else:
            raise ValueError('Unknown command %s' % self.data_type)

        # log
        self.log = data_param.log(self.data_type, self.name)
        self.log.set_log(
            print_frequency=50,
            save_summaries_iter=50,
            save_model_iter=1000,
            test_interval=1000)

        # show
        self._print()

    def _test(self):
        self.batch_size = 64
        self.total_num = 10000
        self.name = self.name + '_test'
        self.reader_thread = 1
        self.shuffle = False
        self.data_path = '../_datasets/cifar10/test.txt'

    def _train(self):
        # basic param
        self.batch_size = 64
        self.total_num = 50000
        self.name = self.name + '_train'
        self.reader_thread = 1
        self.shuffle = True
        self.data_path = '../_datasets/cifar10/train.txt'

        # optimizer
        self.opt = data_param.optimizer()
        self.opt.set_momentum(0.9)

        # lr
        self.lr = data_param.learning_rate()
        self.lr.set_fixed(learning_rate=0.1)
