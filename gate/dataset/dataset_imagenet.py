# -*- coding: utf-8 -*-
""" updated: 2017/04/26
"""

from gate.data import database
from gate.data import data_param
from gate.data import data_loader


class imagenet(database.Database):

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
        self.frames = 1
        self.channels = 3
        self.raw_height = 256
        self.raw_width = 256
        self.output_height = 224
        self.output_width = 224
        self.min_queue_num = 128
        self.device = '/gpu:0'
        self.num_classes = 63
        self.preprocessing_method = 'vgg'

        if data_type == 'train':
            self._train()
        elif data_type == 'test':
            self._test()
        else:
            raise ValueError('Unknown command %s' % self.data_type)

        # log
        self.log = data_param.log(self.data_type, self.name)
        self.log.set_log(
            print_frequency=200,
            save_summaries_iter=200,
            save_model_iter=5000,
            test_interval=5000)

        # show
        self._print()

    def _test(self):
        self.batch_size = 100
        self.total_num = 50000
        self.name = self.name + '_test'
        self.reader_thread = 32
        self.shuffle = False
        self.data_path = '../_datasets/ImageNet/test.txt'

    def _train(self):
        # basic param
        self.batch_size = 32
        self.total_num = 1272104
        self.name = self.name + '_train'
        self.reader_thread = 32
        self.shuffle = True
        self.data_path = '../_datasets/ImageNet/train.txt'

        # optimizer
        self.opt = data_param.optimizer()
        self.opt.set_momentum(0.9)

        # lr
        self.lr = data_param.learning_rate()
        self.lr.set_fixed(learning_rate=0.1)
