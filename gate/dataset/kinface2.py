# -*- coding: utf-8 -*-
""" updated: 2017/3/16
"""

from gate.data import database
from gate.data import data_param
from gate.data import data_loader


class kinface2(database.Database):

    def loads(self):
        # for featrue extract
        # return data_loader.load_image_from_text(
        #     self.data_path, self.shuffle, self.data_type,
        #     self.frames, self.channels, self.preprocessing_method,
        #     self.raw_height, self.raw_width,
        #     self.output_height, self.output_width,
        #     self.min_queue_num, self.batch_size, self.reader_thread)
        return data_loader.load_pair_image_from_text(
            self.data_path, self.shuffle, self.data_type,
            self.frames, self.channels,
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
        self.output_height = 160
        self.output_width = 160
        self.min_queue_num = 256
        self.num_classes = 2
        self.preprocessing_method = 'kinface'
        self.preprocessing_method1 = 'kinface'
        self.preprocessing_method2 = 'kinface'

        # hps
        self.net_name = 'inception_resnet_v1'
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
        self.log = data_param.log(self.data_type, self.name)
        self.log.set_log(
            print_frequency=20,
            save_summaries_iter=20,
            save_model_iter=200,
            test_interval=200)

        # show
        self._print()

    def _test(self):
        self.batch_size = 100
        self.total_num = 500
        self.name = self.name + '_test'
        self.reader_thread = 1
        self.shuffle = False
        self.data_path = '../_datasets/kinface2/md.txt'

    def _train(self):
        # basic param
        self.batch_size = 32
        self.total_num = 400
        self.name = self.name + '_train'
        self.reader_thread = 4
        self.shuffle = True
        self.data_path = '../_datasets/kinface2/fd_train_1.txt'

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


class kinface2_feature(database.Database):

    def loads(self):
        return data_loader.load_pair_numeric_data_from_npy(
            self.data_path, self.shuffle, self.data_type,
            self.min_queue_num, self.batch_size, self.reader_thread)

    def __init__(self, data_type, name):

        self.data_type = data_type
        self.name = name
        self.num_classes = 2
        self.min_queue_num = 64

        # hps
        self.net_name = 'mlp'
        self.hps = data_param.hps(self.net_name)
        self.hps.set_dropout(0.5)
        self.hps.set_weight_decay(0.002)
        # self.hps.set_batch_norm(
        #     batch_norm_decay=0.997,
        #     batch_norm_epsilon=1e-5)

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
        # self._print()

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

        # optimizer
        self.opt = data_param.optimizer()
        self.opt.set_rmsprop()
        # self.opt.set_adam(
        #     adam_beta1=0.9,
        #     adam_beta2=0.999,
        #     adam_epsilon=1e-8)

        # lr
        self.lr = data_param.learning_rate()
        self.lr.set_fixed(learning_rate=0.00005)
