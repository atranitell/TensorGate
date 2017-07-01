# -*- coding: utf-8 -*-
""" updated: 2017/3/16
"""
from gate.data import database
from gate.data import data_param
from gate.data import data_loader


class avec2014_audio(database.Database):

    def loads(self):
        return data_loader.load_continuous_audio_from_npy(
            self.data_path, self.shuffle, self.data_type, self.audio,
            self.min_queue_num, self.batch_size, self.reader_thread)

    def __init__(self, data_type, name, chkp_path=None):
        # init basic class
        database.Database.__init__(self)

        # basic info
        self.data_type = data_type
        self.name = name
        self.num_classes = 63
        self.min_queue_num = 32
        self._set_phase(data_type)

        # audio
        self.audio = data_param.audio()
        # number of steps
        # all frame will be loaded in lstm at once.
        self.audio.frames = 64
        self.audio.frame_length = 200
        self.audio.frame_invl = 200

        # config rnn network
        # self.rnn = data_param.rnn()
        # self.rnn.net_name = 'basic_rnn'
        # self.rnn.activation = 'tanh'
        # self.rnn.cell = 'gru'
        # self.rnn.initializer = 'orthogonal'
        # self.rnn.num_units = 256
        # self.rnn.num_layers = 2
        # self.rnn.dropout = 0.8

        # log
        self.log = data_param.log(data_type, name, chkp_path)
        self.log.set_log(
            print_frequency=10,
            save_summaries_iter=10,
            save_model_iter=1000,
            test_interval=1000)

        # optimizer
        self.opt = data_param.optimizer()
        # self.opt.set_sgd()
        self.opt.set_adam(
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8)
        # self.opt.set_clip_by_value(-1.0, 1.0)

        # lr
        self.lr = data_param.learning_rate()
        self.lr.set_fixed(learning_rate=0.001)

        self._print()

    def _test(self):
        self.batch_size = 32
        # 6327
        self.total_num = 25561
        self.name = self.name + '_test'
        self.reader_thread = 16
        self.shuffle = False
        self.data_path = '../_datasets/AVEC2014_Audio/pp_tst_succ.txt'

    def _train(self):
        self.batch_size = 16
        self.total_num = 160
        self.name = self.name + '_train'
        self.reader_thread = 16
        self.shuffle = True
        self.data_path = '../_datasets/AVEC2014_Audio/pp_trn_0_succ.txt'
