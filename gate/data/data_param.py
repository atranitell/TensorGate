# -*- coding: utf-8 -*-
""" updated: 2017/04/05
    more clearly to setting param
"""

from gate.utils import filesystem


def check_init(_func):
    """ in order to init twice,
        but param will could not be shown
        so it stops to use.
    """

    def _decorate(self, *arg):
        if self.init is True:
            raise ValueError('The class has been initialized!')
        _func(self, *arg)
        self.init = True
    return _decorate


class optimizer():
    """ "adadelta", "adagrad", "adam", "ftrl", 
        "momentum", "sgd", "rmsprops"
    """

    def __init__(self):
        """ The method is only called once.
            Calling again will raise error.
        """
        self.clip_method = None
        # self.init = False

    def set_adam(self, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8):
        self.optimizer = 'adam'
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon

    def set_sgd(self):
        self.optimizer = 'sgd'

    def set_adagrad(self, adagrad_initial_accumulator_value=0.1):
        self.optimizer = 'adagrad'
        self.adagrad_initial_accumulator_value = adagrad_initial_accumulator_value

    def set_adadelta(self, adadelta_epsilon=1e-8, adadelta_rho=0.95):
        self.optimizer = 'adadelta'
        self.adadelta_epsilon = adadelta_epsilon
        self.adadelta_rho = adadelta_rho

    def set_ftrl(self, ftrl_learning_rate_power=-0.5,
                 ftrl_initial_accumulator_value=0.1,
                 ftrl_l1=0.0, ftrl_l2=0.0):
        self.optimizer = 'ftrl'
        self.ftrl_learning_rate_power = ftrl_learning_rate_power
        self.ftrl_initial_accumulator_value = ftrl_initial_accumulator_value
        self.ftrl_l1 = ftrl_l1
        self.ftrl_l2 = ftrl_l2

    def set_momentum(self, momentum=0.9):
        self.optimizer = 'momentum'
        self.momentum = momentum

    def set_rmsprop(self, rmsprop_decay=0.9,
                    rmsprop_momentum=0.0,
                    rmsprop_epsilon=1e-10):
        self.optimizer = 'rmsprop'
        self.rmsprop_decay = rmsprop_decay
        self.rmsprop_momentum = rmsprop_momentum
        self.rmsprop_epsilon = rmsprop_epsilon

    def set_clip_by_value(self, cmin, cmax):
        self.clip_method = 'clip_by_value'
        self.clip_value_min = cmin
        self.clip_value_max = cmax


class learning_rate():
    """ Specifies how the learning rate is decayed. One of "fixed",
        "exponential", or "polynomial"
    """

    def __init__(self, moving_average_decay=None):
        # Whether or not to synchronize the replicas during training.
        self.sync_replicas = False
        # The amount of label smoothing.
        # self.label_smoothing = 0.0
        # The Number of gradients to collect before updating params.
        # self.replicas_to_aggregate = 1
        # The decay to use for the moving average.
        # If left as None, then moving averages are not used.
        self.moving_average_decay = moving_average_decay

    def set_fixed(self, learning_rate=0.1):
        self.learning_rate_decay_type = 'fixed'
        self.learning_rate = learning_rate
        self.num_epochs_per_decay = 999999

    def set_exponential(self, learning_rate=0.1,
                        num_epochs_per_decay=30,
                        learning_rate_decay_factor=0.1):
        self.learning_rate_decay_type = 'exponential'
        self.learning_rate = learning_rate
        self.num_epochs_per_decay = num_epochs_per_decay
        self.learning_rate_decay_factor = learning_rate_decay_factor

    def set_polynomial(self, learning_rate=0.1,
                       num_epochs_per_decay=30,
                       end_learning_rate=0.00001):
        self.learning_rate_decay_type = 'polynomial'
        self.learning_rate = learning_rate
        self.num_epochs_per_decay = num_epochs_per_decay
        self.end_learning_rate = end_learning_rate


class log():

    def __init__(self, data_type, dirname, chkp_path=None):
        if chkp_path is None:
            if data_type == 'train':
                self.train_dir = filesystem.create_folder_with_date(
                    '../_output/' + dirname)
        else:
            if data_type == 'train':
                self.train_dir = chkp_path
            elif data_type == 'test':
                self.test_dir = chkp_path + '/test/'
                filesystem.create_folder(self.test_dir)
            elif data_type == 'val':
                self.val_dir = chkp_path + '/val/'
                filesystem.create_folder(self.val_dir)
            else:
                raise ValueError('Unkonwn data type.')

    def set_log(self, print_frequency=20,
                save_summaries_iter=2,
                save_model_iter=100,
                test_interval=100,
                max_iter=999999):
        # The frequency with which logs are print.
        self.print_frequency = print_frequency
        # The frequency with which summaries are saved, in iteration.
        self.save_summaries_iter = save_summaries_iter
        # The frequency with which the model is saved, in iteration.
        self.save_model_iter = save_model_iter
        # test iteration
        self.test_interval = test_interval
        # max iter to stop
        self.max_iter = max_iter


class hps():

    def __init__(self, net_name):
        self.net_name = net_name
        self.dropout = None
        self.weight_decay = None
        self.batch_norm_decay = None
        self.batch_norm_epsilon = None
        self.batch_norm_scale = None

    def set_dropout(self, dropout=None):
        self.dropout = dropout

    def set_weight_decay(self, weight_decay=None):
        self.weight_decay = weight_decay

    def set_batch_norm(self, batch_norm_decay=None, batch_norm_epsilon=None):
        self.batch_norm_decay = batch_norm_decay
        self.batch_norm_epsilon = batch_norm_epsilon
        self.batch_norm_scale = True


class image():

    def __init__(self):
        self.channels = None
        self.frames = None
        self.raw_height = None
        self.raw_width = None
        self.output_height = None
        self.output_width = None
        self.preprocessing_method1 = None
        self.preprocessing_method2 = None

    def set_format(self, channels, frames=1):
        """ For RGB, channels is 3
            For video input, frames is N else 1 (for single image)
        """
        self.channels = channels
        self.frames = frames

    def set_raw_size(self, height, width):
        """ the size of raw image when loaded in system
        """
        self.raw_height = height
        self.raw_width = width

    def set_output_size(self, height, width):
        """ the order is:
            image to be reshape to raw size
            and output via random crop/center crop
        """
        self.output_height = height
        self.output_width = width

    def set_preprocessing(self, method1=None, method2=None):
        """ method1 for image1, method2 for image2
        """
        self.preprocessing_method1 = method1
        self.preprocessing_method2 = method2


class audio():

    def __init__(self):
        pass


class rnn():

    def __init__(self):
        pass