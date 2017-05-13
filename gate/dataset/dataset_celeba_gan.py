# -*- coding: utf-8 -*-
""" updated: 2017/05/10
"""

from gate.data import database
from gate.data import data_param
from gate.data import data_loader


class celeba_gan(database.Database):

    def loads(self):
        return data_loader.load_image_from_text_multi_label(
            self.data_path, self.shuffle, self.data_type, self.num_classes,
            self.frames, self.channels, self.preprocessing_method,
            self.raw_height, self.raw_width, self.output_height, self.output_width,
            self.min_queue_num, self.batch_size, self.reader_thread)

    def __init__(self, data_type, name):
        self.data_type = data_type
        self.name = name
        self.channels = 3
        self.frames = 1
        self.raw_height = 218
        self.raw_width = 178
        self.output_height = 128
        self.output_width = 128
        self.min_queue_num = 64
        self.device = '/gpu:0'
        self.num_classes = 40  # 40 attribution for face
        self.one_hot = True
        self.preprocessing_method = 'celeba_gan'

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
            save_summaries_iter=20,
            save_model_iter=100,
            test_interval=100)

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
        self.batch_size = 8
        self.total_num = 202599
        self.name = self.name + '_train'
        self.reader_thread = 1
        self.shuffle = False
        self.data_path = '../_datasets/celeba/train.txt'

        # opt
        # specify for different gan
        self.opt = None

        # lr
        self.lr = data_param.learning_rate()
        self.lr.set_fixed(learning_rate=0.00002)