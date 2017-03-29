# -*- coding: utf-8 -*-
""" updated: 2017/3/16
"""

from gate.data import data_model
from gate.utils import filesystem


class avec2014_16f():

    def loads(self):
        """ public interface for upper layer to call
        """
        return data_model.load_single_video_frame(
            self.data_path, self.shuffle, self.data_type, self.channels,
            self.preprocessing_method, self.raw_height, self.raw_width,
            self.output_height, self.output_width,
            self.min_queue_num, self.batch_size, self.reader_thread)

    def __init__(self, data_type, name):
        self.data_type = data_type
        self.name = name

        if data_type == 'train':
            self.batch_size = 2
            self.total_num = 199
            self.name = self.name + '_train'
            self.reader_thread = 32
            self.shuffle = True
            self.data_path = '_datasets/AVEC2014/pp_trn.txt'

        elif data_type == 'test':
            self.batch_size = 1
            self.total_num = 100
            self.name = self.name + '_test'
            self.reader_thread = 1
            self.shuffle = False
            self.data_path = '_datasets/AVEC2014/pp_tst.txt'

        else:
            raise ValueError('Unknown command %s' % self.data_type)

        self.raw_height = 256
        self.raw_width = 256
        self.output_height = 224
        self.output_width = 224
        self.min_queue_num = 128
        self.data_load_method = 'single_video_from_text'
        self.channels = 16
        self.device = '/gpu:0'
        self.num_classes = 63
        self.preprocessing_method = 'avec2014'

        class param():
            pass

        self.log = param()
        # Directory where checkpoints and event logs are written to.
        if self.data_type == 'train':
            self.log.train_dir = filesystem.create_folder_with_date('_output/' + self.name)

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

        """ "adadelta", "adagrad", "adam", "ftrl", "momentum", "sgd", "rmsprops"
        """
        self.opt = param()
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

        """ Specifies how the learning rate is decayed. One of "fixed",
            "exponential", or "polynomial"
        """
        self.lr = param()
        self.lr.learning_rate_decay_type = 'exponential'
        # Initial learning rate.
        self.lr.learning_rate = 0.01
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
