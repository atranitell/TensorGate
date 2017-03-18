# -*- coding: utf-8 -*-
""" updated: 2017/3/16
"""

from data import dataset
from data import utils


class avec2014(dataset.Dataset):

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
            self.log.train_dir = utils.dir_log_constructor('_output/avec2014_train')
        elif self.data_type == 'test':
            self.log.test_dir = None
        # The frequency with which logs are print.
        self.log.print_frequency = 20
        # The frequency with which summaries are saved, in iteration.
        self.log.save_summaries_iter = 20
        # The frequency with which the model is saved, in iteration.
        self.log.save_model_iter = 500
        # test iteration
        self.log.test_interval = 500

    def _init_opt_param(self):
        """The name of the optimizer:
        Args:
            "adadelta", "adagrad", "adam", "ftrl", "momentum", "sgd", "rmsprop"
        """
        class opt_param():
            pass
        self.opt = opt_param()

        self.opt.optimizer = 'momentum'

        """ SGD """
        self.opt.weight_decay = 0.0005
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
        self.lr.learning_rate = 0.01
        # The minimal end learning rate used by a polynomial decay learning
        # rate.
        self.lr.end_learning_rate = 0.00001
        # The amount of label smoothing.
        self.lr.label_smoothing = 0.0
        # Learning rate decay factor
        self.lr.learning_rate_decay_factor = 0.94
        # Number of epochs after which learning rate decays.
        self.lr.num_epochs_per_decay = 2.0
        # Whether or not to synchronize the replicas during training.
        self.lr.sync_replicas = False
        # The Number of gradients to collect before updating params.
        self.lr.replicas_to_aggregate = 1
        # The decay to use for the moving average.
        # If left as None, then moving averages are not used.
        self.lr.moving_average_decay = None

    def _init_common_param(self):
        self.batch_size = 32
        self.output_height = 224
        self.output_width = 224
        self.min_queue_num = 128
        self.device = '/gpu:0'
        self.num_classes = 100
        self.preprocessing_method = 'vgg'

    def _init_train_param(self):
        self.total_num = 15660
        self.name = 'avec2014_train'
        self.reader_thread = 8
        self.shuffle = True
        self.data_load_method = 'text'  # 'text' / 'tfrecord'
        self.data_path = '_datasets/AVEC2014/trn_list.txt'

    def _init_test_param(self):
        self.total_num = 17727
        self.name = 'avec2014_test'
        self.reader_thread = 8
        self.shuffle = False
        self.data_load_method = 'text'
        self.data_path = '_datasets/AVEC2014/tst_list.txt'

    def loads(self):
        """ load images and labels from folder/files."""
        # load from disk
        image, label, filename = self._load_data(
            self.data_load_method, self.data_path, self.total_num, self.shuffle)

        # preprocessing batch size
        image = self._preprocessing_image(self.preprocessing_method, self.data_type, image,
                                          self.output_height, self.output_width)

        # preprocessing images
        label = self._preprocessing_label(label, self.data_type)

        # return [images, labels, filenames] as a batch
        return self._generate_image_label_batch(image, label, self.shuffle, self.min_queue_num,
                                                self.batch_size, self.reader_thread, filename)
