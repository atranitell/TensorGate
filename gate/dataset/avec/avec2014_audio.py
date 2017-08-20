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
        self.min_queue_num = 16
        self._set_phase(data_type)

        # audio
        self.audio = data_param.audio()
        # number of steps
        # all frame will be loaded in lstm at once.
        self.audio.frames = 64
        self.audio.frame_length = 200
        self.audio.frame_invl = 200

        # setting hps
        self.hps = data_param.hps('audionet')
        self.hps.set_dropout(0.5)
        self.hps.set_weight_decay(0.0001)
        self.hps.set_batch_norm(
            batch_norm_decay=0.999,
            batch_norm_epsilon=1e-5,
            batch_norm_scale=False)

        # log
        self.log = data_param.log(data_type, name, chkp_path)
        self.log.set_log(
            print_frequency=50,
            save_summaries_iter=50,
            save_model_iter=1000,
            test_interval=2000)

        # optimizer
        self.opt = data_param.optimizer()
        self.opt.set_adam(
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8)

        # lr
        self.lr = data_param.learning_rate()
        self.lr.set_fixed(learning_rate=0.0001)

        self._print()

    def _train(self):
        self.batch_size = 32
        self.total_num = 100
        self.name = self.name + '_train'
        self.reader_thread = 16
        self.shuffle = True
        self.data_path = '../_datasets/AVEC2014_Audio/pp_trn_raw.txt'

    def _val_train(self):
        self.batch_size = 50
        # 32-30718 # 64-15292 # 128-7571
        self.total_num = 7571
        self.name = self.name + '_val_train'
        self.reader_thread = 16
        self.shuffle = False
        self.data_path = '../_datasets/AVEC2014_Audio/pp_trn_succ128.txt'

    def _val(self):
        self.batch_size = 50
        # 32-28542 # 64-14199 # 128-7027
        self.total_num = 27942
        self.name = self.name + '_val'
        self.reader_thread = 16
        self.shuffle = False
        self.data_path = '../_datasets/AVEC2014_Audio/pp_val_over128.txt'

    def _test(self):
        self.batch_size = 50
        # 32-51074 # 64-25465 # 128-12661
        self.total_num = 50474
        self.name = self.name + '_test'
        self.reader_thread = 16
        self.shuffle = False
        self.data_path = '../_datasets/AVEC2014_Audio/pp_tst_over128.txt'
