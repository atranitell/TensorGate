# -*- coding: utf-8 -*-
""" updated: 2017/3/16
"""
import os
import random
import math
import numpy as np
from PIL import Image

import tensorflow as tf
from data import dataset
from data import utils


class avec2014_flow_16f(dataset.Dataset):

    def __init__(self, data_type):
        self.data_type = data_type

        if data_type == 'train':
            self._init_train_param()
        elif data_type == 'test':
            self._init_test_param()
        else:
            raise ValueError('Unknown command %s' % self.data_type)

        self._init_common_param()
        self._init_lrn_param()
        self._init_opt_param()
        self._init_log_param()

    def _init_log_param(self):
        class log_param():
            pass
        self.log = log_param()
        # Directory where checkpoints and event logs are written to.
        if self.data_type == 'train':
            self.log.train_dir = utils.dir_log_constructor('_output/avec2014_flow_16f_train')
        elif self.data_type == 'test':
            self.log.test_dir = None
        # The frequency with which logs are print.
        self.log.print_frequency = 20
        # The frequency with which summaries are saved, in iteration.
        self.log.save_summaries_iter = 2
        # The frequency with which the model is saved, in iteration.
        self.log.save_model_iter = 200
        # test iteration
        self.log.test_interval = 200

    def _init_opt_param(self):
        """The name of the optimizer: 
        Args: 
            "adadelta", "adagrad", "adam", "ftrl", "momentum", "sgd", "rmsprops"
        """
        class opt_param():
            pass
        self.opt = opt_param()

        self.opt.optimizer = 'adam'

        """ SGD """
        self.opt.weight_decay = 0.0001
        self.opt.momentum = 0.9
        self.opt.opt_epsilon = 1.0

        """ ADAGRAD """
        self.opt.adadelta_rho = 0.95
        self.opt.adagrad_initial_accumulator_value = 0.1

        """ ADAMs """
        self.opt.adam_beta1 = 0.9
        self.opt.adam_beta2 = 0.999

        """ FTRL """
        self.opt.ftrl_learning_rate_power = -0.5
        self.opt.ftrl_initial_accumulator_value = 0.1
        self.opt.ftrl_l1 = 0.0
        self.opt.ftrl_l2 = 0.0

        """ RMSProp """
        self.opt.rmsprop_momentum = 0.9
        # Decay term for RMSProp.
        self.opt.rmsprop_decay = 0.9

    def _init_lrn_param(self):
        class opt():
            pass
        self.lr = opt()
        # Specifies how the learning rate is decayed. One of "fixed",
        # "exponential", or "polynomial"
        self.lr.learning_rate_decay_type = 'exponential'
        # Initial learning rate.
        self.lr.learning_rate = 0.1
        # The minimal end learning rate used by a polynomial decay learning
        # rate.
        self.lr.end_learning_rate = 0.00001
        # The amount of label smoothing.
        self.lr.label_smoothing = 0.0
        # Learning rate decay factor
        self.lr.learning_rate_decay_factor = 0.5
        # Number of epochs after which learning rate decays.
        self.lr.num_epochs_per_decay = 5000.0
        # Whether or not to synchronize the replicas during training.
        self.lr.sync_replicas = False
        # The Number of gradients to collect before updating params.
        self.lr.replicas_to_aggregate = 1
        # The decay to use for the moving average.
        # If left as None, then moving averages are not used.
        self.lr.moving_average_decay = None

    def _init_common_param(self):
        self.output_height = 224
        self.output_width = 224
        self.min_queue_num = 128
        self.device = '/gpu:0'
        self.num_classes = 63
        self.preprocessing_method = ''

    def _init_train_param(self):
        self.batch_size = 32
        self.total_num = 199
        self.name = 'avec2014_flow_16f_train'
        self.reader_thread = 16
        self.shuffle = True
        self.data_load_method = 'text'
        self.data_path = '_datasets/AVEC2014/pp_trn_flow.txt'

    def _init_test_param(self):
        self.batch_size = 1
        self.total_num = 100
        self.name = 'avec2014_flow_16f_test'
        self.reader_thread = 1
        self.shuffle = False
        self.data_load_method = 'text'
        self.data_path = '_datasets/AVEC2014/pp_tst_flow.txt'

    def load_from_files(self):
        """ Load data from file """
        fold_list = []
        label_list = []

        with open(self.data_path, 'r') as fp:
            for line in fp:
                r = line.split(' ')
                if len(r) <= 1:
                    continue
                fold_list.append(r[0])
                label_list.append(int(r[1]))

        folds = tf.convert_to_tensor(fold_list, dtype=tf.string)
        labels = tf.convert_to_tensor(label_list, dtype=tf.int32)

        return folds, labels

    def load_images_from_fold(self, foldpath):
        # default choose 16
        channels = 16
        foldpath_str = str(foldpath, encoding='utf-8')
        fold_path_abs = os.path.join('_datasets/AVEC2014', foldpath_str)
        img_list = []
        for fs in os.listdir(fold_path_abs):
            if len(fs.split('.jpg')) > 1:
                img_list.append(fs)

        # generate idx without reptitious
        # img_indice = random.sample([i for i in range(len(img_list))], channels)
        invl = math.floor(len(img_list) / float(channels))
        start = 0
        img_indice = []

        for _ in range(channels):
            end = start + invl
            if self.data_type == 'train':
                img_indice.append(random.randint(start, end-1))
            elif self.data_type == 'test':
                img_indice.append(start)
            start = end

        # generate
        img_selected_list = []
        for idx in range(channels):
            img_path = os.path.join(fold_path_abs, img_list[img_indice[idx]])
            img_selected_list.append(img_path)
        img_selected_list.sort()

        # for line in img_selected_list:
        #     print(line)
        # raise ValueError(123)

        # compression to (256,256,3*16)
        combine = np.asarray(Image.open(img_selected_list[0]))
        for idx, img in enumerate(img_selected_list):
            if idx == 0:
                continue
            img_content = np.asarray(Image.open(img))
            combine = np.dstack((combine, img_content))
        return combine

    def loads(self):
        """ load images and labels from folder/files."""
        # load from disk
        folds, labels = self.load_from_files()
        # construct a fifo queue
        foldname, label = tf.train.slice_input_producer([folds, labels], shuffle=self.shuffle)

        image = tf.py_func(self.load_images_from_fold, [foldname], tf.uint8)
        image = tf.reshape(image, shape=[256, 256, 48])

        if self.data_type == 'train':
            out_image = tf.random_crop(image, [self.output_height, self.output_width, 48])
            # out_image = tf.image.random_flip_left_right(distorted_image)
            # distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
            # distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
        elif self.data_type == 'test':
            out_image = tf.image.resize_image_with_crop_or_pad(
                image, self.output_height, self.output_width)
            # tf.summary.image('resized_image', tf.expand_dims(resized_image, 0))

        # Subtract off the mean and divide by the variance of the pixels.
        image = tf.image.per_image_standardization(out_image)

        # preprocessing images
        label = self._preprocessing_label(label, self.data_type)

        # return [images, labels, filenames] as a batch
        return self._generate_image_label_batch(image, label, self.shuffle, self.min_queue_num,
                                                self.batch_size, self.reader_thread, foldname)
