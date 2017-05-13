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
        pass
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


class learning_rate():
    """ Specifies how the learning rate is decayed. One of "fixed",
        "exponential", or "polynomial"
    """

    def __init__(self):
        # Whether or not to synchronize the replicas during training.
        self.sync_replicas = False
        # The amount of label smoothing.
        # self.label_smoothing = 0.0
        # The Number of gradients to collect before updating params.
        # self.replicas_to_aggregate = 1
        # The decay to use for the moving average.
        # If left as None, then moving averages are not used.
        # self.moving_average_decay = None
        # init
        # self.init = False

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

    def __init__(self, data_type, dirname):
        if data_type == 'train':
            self.train_dir = filesystem.create_folder_with_date(
                '../_output/' + dirname)
        # it will be determined by command inputs.
        elif data_type == 'test':
            self.test_dir = None
        elif data_type == 'val':
            self.val_dir = None
        else:
            raise ValueError('Unkonwn data type.')

    def set_log(self, print_frequency=20,
                save_summaries_iter=2,
                save_model_iter=100,
                test_interval=100):
        # The frequency with which logs are print.
        self.print_frequency = print_frequency

        # The frequency with which summaries are saved, in iteration.
        self.save_summaries_iter = save_summaries_iter

        # The frequency with which the model is saved, in iteration.
        self.save_model_iter = save_model_iter

        # test iteration
        self.test_interval = test_interval
